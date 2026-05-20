from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.dataset import _to_tensor_opt, _to_tensor_sar, load_pair_images
from src.data.pairs import PairRecord, read_pairs_csv
from src.models import build_model
from src.utils.metrics import count_parameters


def _parse_int_tuple(value: str | None) -> tuple[int, ...]:
    if value is None:
        return (64, 128, 256)
    return tuple(int(v.strip()) for v in value.split(",") if v.strip())


def _parse_float_tuple(value: str | None) -> tuple[float, ...] | None:
    if value is None or not value.strip():
        return None
    return tuple(float(v.strip()) for v in value.split(",") if v.strip())


def _crop(arr: np.ndarray, cx: int, cy: int, size: int) -> np.ndarray | None:
    half = size // 2
    x0, y0 = cx - half, cy - half
    x1, y1 = x0 + size, y0 + size
    if x0 < 0 or y0 < 0 or x1 > arr.shape[1] or y1 > arr.shape[0]:
        return None
    if arr.ndim == 2:
        return arr[y0:y1, x0:x1]
    return arr[y0:y1, x0:x1, :]


def _grid_centers(width: int, height: int, patch_size: int, step: int) -> np.ndarray:
    half = patch_size // 2
    xs = range(half, width - half + 1, step)
    ys = range(half, height - half + 1, step)
    return np.asarray([[x, y] for y in ys for x in xs], dtype=np.float32)


def _extract_features(
    model,
    image: np.ndarray,
    centers: np.ndarray,
    patch_size: int,
    modality: str,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    tensors: list[torch.Tensor] = []
    features: list[torch.Tensor] = []
    encode = model.encode_sar if modality == "sar" else model.encode_opt
    to_tensor = _to_tensor_sar if modality == "sar" else _to_tensor_opt
    with torch.no_grad():
        for cx, cy in centers.astype(int):
            patch = _crop(image, int(cx), int(cy), patch_size)
            if patch is None:
                continue
            tensors.append(to_tensor(patch))
            if len(tensors) == batch_size:
                batch = torch.stack(tensors).to(device)
                features.append(encode(batch).cpu())
                tensors.clear()
        if tensors:
            batch = torch.stack(tensors).to(device)
            features.append(encode(batch).cpu())
    if not features:
        return torch.empty((0, 1), dtype=torch.float32)
    return torch.cat(features, dim=0)


def _refine_with_softmax(coords: np.ndarray, sims: np.ndarray, top_k: int, temperature: float) -> tuple[np.ndarray, float]:
    if len(coords) == 0:
        return np.asarray([np.nan, np.nan], dtype=np.float32), 0.0
    k = min(top_k, len(coords))
    idx = np.argpartition(sims, -k)[-k:]
    local_scores = sims[idx]
    local_coords = coords[idx]
    weights = np.exp((local_scores - local_scores.max()) / max(temperature, 1e-6))
    weights = weights / max(weights.sum(), 1e-12)
    return (local_coords * weights[:, None]).sum(axis=0).astype(np.float32), float(local_scores.max())


def _weighted_affine(src: np.ndarray, dst: np.ndarray, weights: np.ndarray) -> np.ndarray | None:
    if len(src) < 3:
        return None
    weights = np.sqrt(np.clip(weights, 1e-6, None))
    rows = []
    values = []
    for (x, y), (u, v), w in zip(src, dst, weights):
        rows.append([w * x, w * y, w, 0.0, 0.0, 0.0])
        rows.append([0.0, 0.0, 0.0, w * x, w * y, w])
        values.extend([w * u, w * v])
    matrix, *_ = np.linalg.lstsq(np.asarray(rows, dtype=np.float64), np.asarray(values, dtype=np.float64), rcond=None)
    return matrix.reshape(2, 3).astype(np.float32)


def _match_scale(
    model,
    sar: np.ndarray,
    opt: np.ndarray,
    patch_size: int,
    step: int,
    search_radius: float,
    threshold: float,
    top_k: int,
    temperature: float,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, w = sar.shape
    sar_centers = _grid_centers(w, h, patch_size, step)
    opt_centers = _grid_centers(w, h, patch_size, step)
    sar_features = _extract_features(model, sar, sar_centers, patch_size, "sar", device, batch_size)
    opt_features = _extract_features(model, opt, opt_centers, patch_size, "opt", device, batch_size)
    correlation = (sar_features @ opt_features.T).numpy()

    src_pts: list[np.ndarray] = []
    dst_pts: list[np.ndarray] = []
    scores: list[float] = []
    levels: list[int] = []
    for row, center in enumerate(sar_centers):
        distances = np.linalg.norm(opt_centers - center[None, :], axis=1)
        mask = distances <= search_radius
        if not mask.any():
            continue
        local_coords = opt_centers[mask]
        local_sims = correlation[row, mask]
        refined, score = _refine_with_softmax(local_coords, local_sims, top_k=top_k, temperature=temperature)
        if score >= threshold and np.isfinite(refined).all():
            src_pts.append(center)
            dst_pts.append(refined)
            scores.append(score)
            levels.append(patch_size)
    return (
        np.asarray(src_pts, dtype=np.float32),
        np.asarray(dst_pts, dtype=np.float32),
        np.asarray(scores, dtype=np.float32),
        np.asarray(levels, dtype=np.int32),
    )


def estimate_g10_matches(
    model,
    record: PairRecord,
    device: torch.device,
    region_size: int,
    block_size: int,
    region_step: int,
    block_step: int,
    search_radius: int,
    threshold: float,
    top_k: int,
    temperature: float,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sar, opt = load_pair_images(record)
    all_src, all_dst, all_scores, all_levels = [], [], [], []
    for patch_size, step in ((region_size, region_step), (block_size, block_step)):
        src, dst, scores, levels = _match_scale(
            model,
            sar,
            opt,
            patch_size=patch_size,
            step=step,
            search_radius=float(search_radius),
            threshold=threshold,
            top_k=top_k,
            temperature=temperature,
            device=device,
            batch_size=batch_size,
        )
        if len(src):
            all_src.append(src)
            all_dst.append(dst)
            all_scores.append(scores)
            all_levels.append(levels)
    if not all_src:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )
    return np.vstack(all_src), np.vstack(all_dst), np.concatenate(all_scores), np.concatenate(all_levels)


def main() -> None:
    parser = argparse.ArgumentParser(description="G10 HSPM+DABM correlation registration.")
    parser.add_argument("--pairs", type=Path, default=Path("dl/outputs/pairs.csv"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--sample-id", type=int, required=True)
    parser.add_argument("--region-size", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--region-step", type=int, default=256)
    parser.add_argument("--block-step", type=int, default=128)
    parser.add_argument("--search-radius", type=int, default=192)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    records = read_pairs_csv(args.pairs)
    record = next((r for r in records if r.sample_id == args.sample_id), None)
    if record is None:
        raise SystemExit(f"sample id {args.sample_id} not found in pair index")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_name = args.model or checkpoint.get("model_name")
    if model_name != "g10_hspm_dabm":
        raise SystemExit(f"register_g10_hspm.py requires g10_hspm_dabm checkpoint, got {model_name}")
    ckpt_args = checkpoint.get("args", {})
    device = torch.device(args.device)
    model = build_model(
        model_name,
        dropout=float(ckpt_args.get("dropout", 0.2)),
        cbam_reduction=int(ckpt_args.get("cbam_reduction", 16)),
        attention_placement=ckpt_args.get("attention_placement", "layer4"),
        final_backbone=ckpt_args.get("final_backbone", "resnet34"),
        patch_sizes=_parse_int_tuple(ckpt_args.get("patch_sizes", "64,128,256")),
        fusion_weights=_parse_float_tuple(ckpt_args.get("fusion_weights")),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    start = time.perf_counter()
    src, dst, scores, levels = estimate_g10_matches(
        model,
        record,
        device,
        region_size=args.region_size,
        block_size=args.block_size,
        region_step=args.region_step,
        block_step=args.block_step,
        search_radius=args.search_radius,
        threshold=args.threshold,
        top_k=args.top_k,
        temperature=args.temperature,
        batch_size=args.batch_size,
    )
    if len(src) >= 3:
        ransac_matrix, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        inlier_mask = inliers.ravel().astype(bool) if inliers is not None else np.zeros(len(src), dtype=bool)
        matrix = _weighted_affine(src[inlier_mask], dst[inlier_mask], scores[inlier_mask]) if inlier_mask.any() else ransac_matrix
        if matrix is None:
            matrix = ransac_matrix
        residual = dst[inlier_mask] - cv2.transform(src[inlier_mask][None, :, :], matrix)[0] if matrix is not None and inlier_mask.any() else np.empty((0, 2))
        rmse = float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))) if residual.size else None
    else:
        matrix, inlier_mask, rmse = None, np.zeros(len(src), dtype=bool), None

    result = {
        "sample_id": record.sample_id,
        "stem": record.stem,
        "method": "G10 HSPM+DABM correlation registration",
        "candidate_matches": int(len(src)),
        "ncm": int(len(src)),
        "cmr": float(inlier_mask.mean()) if len(inlier_mask) else 0.0,
        "rmse": rmse,
        "affine_matrix": matrix.tolist() if matrix is not None else None,
        "runtime_sec": time.perf_counter() - start,
        "parameter_count": count_parameters(model),
        "mean_score": float(scores.mean()) if len(scores) else None,
        "region_matches": int((levels == args.region_size).sum()) if len(levels) else 0,
        "block_matches": int((levels == args.block_size).sum()) if len(levels) else 0,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
