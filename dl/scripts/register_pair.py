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


def estimate_matches(
    model,
    record: PairRecord,
    device: torch.device,
    patch_size: int = 128,
    grid_step: int = 128,
    search_radius: int = 32,
    threshold: float = 0.7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sar, opt = load_pair_images(record)
    h, w = sar.shape
    src_pts: list[list[float]] = []
    dst_pts: list[list[float]] = []
    scores: list[float] = []
    offsets = [(0, 0)]
    for d in range(grid_step // 4, search_radius + 1, max(grid_step // 4, 1)):
        offsets.extend([(d, 0), (-d, 0), (0, d), (0, -d), (d, d), (-d, d), (d, -d), (-d, -d)])

    model.eval()
    with torch.no_grad():
        for cy in range(patch_size // 2, h - patch_size // 2, grid_step):
            for cx in range(patch_size // 2, w - patch_size // 2, grid_step):
                sar_patch = _crop(sar, cx, cy, patch_size)
                if sar_patch is None:
                    continue
                best_score = -1.0
                best_xy = None
                sar_t = _to_tensor_sar(sar_patch).unsqueeze(0).to(device)
                for dx, dy in offsets:
                    opt_patch = _crop(opt, cx + dx, cy + dy, patch_size)
                    if opt_patch is None:
                        continue
                    opt_t = _to_tensor_opt(opt_patch).unsqueeze(0).to(device)
                    score = float(torch.sigmoid(model(sar_t, opt_t)).item())
                    if score > best_score:
                        best_score = score
                        best_xy = (cx + dx, cy + dy)
                if best_xy and best_score >= threshold:
                    src_pts.append([cx, cy])
                    dst_pts.append([best_xy[0], best_xy[1]])
                    scores.append(best_score)
    return np.asarray(src_pts, np.float32), np.asarray(dst_pts, np.float32), np.asarray(scores, np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, default=Path("dl/outputs/pairs.csv"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--sample-id", type=int, required=True)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--grid-step", type=int, default=128)
    parser.add_argument("--search-radius", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    records = read_pairs_csv(args.pairs)
    record = next((r for r in records if r.sample_id == args.sample_id), None)
    if record is None:
        raise SystemExit(f"sample id {args.sample_id} not found in pair index")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_name = args.model or checkpoint.get("model_name")
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

    start = time.perf_counter()
    src, dst, scores = estimate_matches(
        model,
        record,
        device,
        patch_size=args.patch_size,
        grid_step=args.grid_step,
        search_radius=args.search_radius,
        threshold=args.threshold,
    )
    if len(src) >= 3:
        matrix, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        inlier_mask = inliers.ravel().astype(bool) if inliers is not None else np.zeros(len(src), dtype=bool)
        residual = dst[inlier_mask] - cv2.transform(src[inlier_mask][None, :, :], matrix)[0] if matrix is not None and inlier_mask.any() else np.empty((0, 2))
        rmse = float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))) if residual.size else None
    else:
        matrix, inlier_mask, rmse = None, np.zeros(len(src), dtype=bool), None

    result = {
        "sample_id": record.sample_id,
        "stem": record.stem,
        "candidate_matches": int(len(src)),
        "ncm": int(len(src)),
        "cmr": float(inlier_mask.mean()) if len(inlier_mask) else 0.0,
        "rmse": rmse,
        "affine_matrix": matrix.tolist() if matrix is not None else None,
        "runtime_sec": time.perf_counter() - start,
        "parameter_count": count_parameters(model),
        "mean_score": float(scores.mean()) if len(scores) else None,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
