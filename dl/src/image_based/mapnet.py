from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
from torch import nn
from torchvision import models


@dataclass
class MapNetResult:
    sample_id: int
    stem: str
    ncm: int
    cmr: float
    rmse: float | None
    affine_matrix: list[list[float]] | None
    runtime_sec: float
    parameter_count: int
    feature_points_sar: int
    feature_points_optical: int


class ResNet50Conv4(nn.Module):
    """ResNet50 feature extractor ending at conv4/layer3."""

    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        base = models.resnet50(weights=weights)
        self.net = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
        )
        self.out_channels = 1024

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def prepare_inputs(sar: np.ndarray, opt: np.ndarray, max_size: int) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    h, w = sar.shape
    scale = min(max_size / max(h, w), 1.0)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    if scale < 1.0:
        sar = cv2.resize(sar, (new_w, new_h), interpolation=cv2.INTER_AREA)
        opt = cv2.resize(opt, (new_w, new_h), interpolation=cv2.INTER_AREA)

    sar_f = sar.astype(np.float32) / 255.0
    opt_f = opt.astype(np.float32) / 255.0
    sar_rgb = np.repeat(sar_f[None, :, :], 3, axis=0)
    opt_chw = opt_f.transpose(2, 0, 1)
    return (
        torch.from_numpy(sar_rgb).unsqueeze(0),
        torch.from_numpy(opt_chw).unsqueeze(0),
        w / new_w,
        h / new_h,
    )


def spap(features: torch.Tensor) -> torch.Tensor:
    """Spatial pyramid average pooling with 4x4, 2x2 and 1x1 branches."""

    _, _, h, w = features.shape
    branches = [features]
    for size in (4, 2, 1):
        pooled = F.adaptive_avg_pool2d(features, output_size=(size, size))
        up = F.interpolate(pooled, size=(h, w), mode="bilinear", align_corners=False)
        branches.append(up)
    fused = torch.cat(branches, dim=1)
    return F.normalize(fused, dim=1)


def attention_select(
    features: torch.Tensor,
    original_width: int,
    original_height: int,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select high-response dense feature locations and return descriptors plus image coords."""

    fmap = features.squeeze(0)
    c, h, w = fmap.shape
    saliency = torch.linalg.vector_norm(fmap, ord=2, dim=0)
    k = min(max_points, h * w)
    flat_idx = torch.topk(saliency.flatten(), k=k).indices.cpu().numpy()
    ys = flat_idx // w
    xs = flat_idx % w
    desc = fmap[:, ys, xs].T.detach().cpu().numpy().astype(np.float32)
    coords = np.stack(
        [
            (xs.astype(np.float32) + 0.5) * (original_width / float(w)),
            (ys.astype(np.float32) + 0.5) * (original_height / float(h)),
        ],
        axis=1,
    )
    return desc, coords


def pca_reduce(a: np.ndarray, b: np.ndarray, dim: int) -> tuple[np.ndarray, np.ndarray]:
    n_components = min(dim, a.shape[1], a.shape[0] + b.shape[0] - 1)
    if n_components <= 0:
        return a, b
    pca = PCA(n_components=n_components, whiten=True, random_state=42)
    both = pca.fit_transform(np.vstack([a, b]))
    both = both.astype(np.float32)
    a_red = both[: len(a)]
    b_red = both[len(a) :]
    a_red /= np.linalg.norm(a_red, axis=1, keepdims=True) + 1e-8
    b_red /= np.linalg.norm(b_red, axis=1, keepdims=True) + 1e-8
    return a_red, b_red


def kd_tree_match(
    sar_desc: np.ndarray,
    opt_desc: np.ndarray,
    sar_coords: np.ndarray,
    opt_coords: np.ndarray,
    ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(sar_desc) == 0 or len(opt_desc) < 2:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
    tree = KDTree(opt_desc)
    dist, ind = tree.query(sar_desc, k=2)
    keep = dist[:, 0] < ratio * (dist[:, 1] + 1e-8)
    src = sar_coords[keep]
    dst = opt_coords[ind[keep, 0]]
    return src.astype(np.float32), dst.astype(np.float32)


def estimate_affine(src: np.ndarray, dst: np.ndarray, threshold: float) -> tuple[np.ndarray | None, np.ndarray, float | None]:
    if len(src) < 3:
        return None, np.zeros(len(src), dtype=bool), None
    matrix, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=threshold)
    if matrix is None or inliers is None:
        return matrix, np.zeros(len(src), dtype=bool), None
    mask = inliers.ravel().astype(bool)
    if not mask.any():
        return matrix, mask, None
    pred = cv2.transform(src[mask][None, :, :], matrix)[0]
    residual = dst[mask] - pred
    rmse = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
    return matrix, mask, rmse


def run_mapnet_like(
    sample_id: int,
    stem: str,
    sar: np.ndarray,
    opt: np.ndarray,
    device: torch.device,
    max_size: int = 512,
    max_points: int = 600,
    pca_dim: int = 64,
    ratio: float = 0.95,
    ransac_thresh: float = 5.0,
    pretrained: bool = False,
) -> MapNetResult:
    start = time.perf_counter()
    model = ResNet50Conv4(pretrained=pretrained).to(device).eval()
    params = parameter_count(model)
    original_height, original_width = sar.shape
    sar_t, opt_t, _, _ = prepare_inputs(sar, opt, max_size=max_size)
    sar_t = sar_t.to(device)
    opt_t = opt_t.to(device)

    with torch.no_grad():
        sar_feat = spap(model(sar_t))
        opt_feat = spap(model(opt_t))

    sar_desc, sar_coords = attention_select(sar_feat, original_width, original_height, max_points)
    opt_desc, opt_coords = attention_select(opt_feat, original_width, original_height, max_points)
    sar_desc, opt_desc = pca_reduce(sar_desc, opt_desc, pca_dim)
    src, dst = kd_tree_match(sar_desc, opt_desc, sar_coords, opt_coords, ratio)
    matrix, inlier_mask, rmse = estimate_affine(src, dst, ransac_thresh)

    return MapNetResult(
        sample_id=sample_id,
        stem=stem,
        ncm=int(len(src)),
        cmr=float(inlier_mask.mean()) if len(inlier_mask) else 0.0,
        rmse=rmse,
        affine_matrix=matrix.tolist() if matrix is not None else None,
        runtime_sec=time.perf_counter() - start,
        parameter_count=params,
        feature_points_sar=int(len(sar_desc)),
        feature_points_optical=int(len(opt_desc)),
    )
