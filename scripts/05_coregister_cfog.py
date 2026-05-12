from __future__ import annotations

import argparse
import datetime as dt
import shlex
from pathlib import Path

import cv2
import numpy as np
import rasterio

from common import ensure_parent, read_first_band_uint8, read_rgb_as_gray_uint8, write_json


def enhance_for_structure(image: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    blur = cv2.GaussianBlur(image, (0, 0), 1.0)
    return clahe.apply(blur)


def cfog_channels(image: np.ndarray, orientations: int, sigma: float) -> np.ndarray:
    img = image.astype(np.float32) / 255.0
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    angle = np.mod(cv2.phase(gx, gy, angleInDegrees=False), np.pi)

    channels = []
    for theta in np.linspace(0, np.pi, orientations, endpoint=False):
        response = np.maximum(np.cos(angle - theta), 0.0) * mag
        response = cv2.GaussianBlur(response, (0, 0), sigma)
        channels.append(response)
    return np.stack(channels, axis=-1)


def detect_points(image: np.ndarray, max_points: int, quality: float, min_distance: float, border: int) -> list[cv2.KeyPoint]:
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.normalize(cv2.magnitude(gx, gy), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    pts = cv2.goodFeaturesToTrack(
        mag,
        maxCorners=max_points,
        qualityLevel=quality,
        minDistance=min_distance,
        blockSize=5,
        useHarrisDetector=True,
        k=0.04,
    )
    if pts is None:
        return []

    keypoints = []
    h, w = image.shape
    for p in pts.reshape(-1, 2):
        x, y = float(p[0]), float(p[1])
        if border <= x < w - border and border <= y < h - border:
            keypoints.append(cv2.KeyPoint(x, y, float(border)))
    return keypoints


def describe_keypoints(channels: np.ndarray, keypoints: list[cv2.KeyPoint], patch: int, cells: int) -> tuple[list[cv2.KeyPoint], np.ndarray]:
    radius = patch // 2
    cell = patch // cells
    h, w, c = channels.shape
    valid_kp = []
    descriptors = []

    for kp in keypoints:
        x = int(round(kp.pt[0]))
        y = int(round(kp.pt[1]))
        if x - radius < 0 or y - radius < 0 or x + radius > w or y + radius > h:
            continue

        roi = channels[y - radius : y + radius, x - radius : x + radius, :]
        feat = []
        for yy in range(cells):
            for xx in range(cells):
                block = roi[yy * cell : (yy + 1) * cell, xx * cell : (xx + 1) * cell, :]
                feat.extend(block.mean(axis=(0, 1)).tolist())

        desc = np.asarray(feat, dtype=np.float32)
        norm = np.linalg.norm(desc)
        if norm > 1e-6:
            desc /= norm
            valid_kp.append(kp)
            descriptors.append(desc)

    if not descriptors:
        return [], np.empty((0, cells * cells * c), dtype=np.float32)
    return valid_kp, np.vstack(descriptors).astype(np.float32)


def match_descriptors(desc_moving: np.ndarray, desc_reference: np.ndarray, ratio: float, mutual: bool) -> list[cv2.DMatch]:
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = matcher.knnMatch(desc_moving, desc_reference, k=2)
    good = []
    for pair in pairs:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio * n.distance:
                good.append(m)

    if not mutual:
        return good

    reverse = matcher.match(desc_reference, desc_moving)
    reverse_best = {m.queryIdx: m.trainIdx for m in reverse}
    return [m for m in good if reverse_best.get(m.trainIdx) == m.queryIdx]


def spatial_gate(matches: list[cv2.DMatch], kp_moving: list[cv2.KeyPoint], kp_reference: list[cv2.KeyPoint], max_displacement: float) -> list[cv2.DMatch]:
    if max_displacement <= 0:
        return matches
    kept = []
    for m in matches:
        x1, y1 = kp_moving[m.queryIdx].pt
        x2, y2 = kp_reference[m.trainIdx].pt
        if np.hypot(x2 - x1, y2 - y1) <= max_displacement:
            kept.append(m)
    return kept


def estimate_affine(src_pts: np.ndarray, dst_pts: np.ndarray, model: str, thresh: float):
    if model == "affine":
        matrix, inliers = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=thresh)
    elif model == "partial-affine":
        matrix, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=thresh)
    else:
        raise ValueError(f"Unsupported model: {model}")
    return matrix, inliers


def rmse_affine(src_pts: np.ndarray, dst_pts: np.ndarray, matrix: np.ndarray, inliers: np.ndarray) -> float:
    mask = inliers.ravel().astype(bool)
    if not np.any(mask):
        return float("nan")
    pred = cv2.transform(src_pts[mask].reshape(-1, 1, 2), matrix).reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum((pred - dst_pts[mask]) ** 2, axis=1))))


def warp_full_resolution(moving_path: Path, reference_path: Path, matrix: np.ndarray, output_path: Path) -> None:
    with rasterio.open(moving_path) as moving, rasterio.open(reference_path) as reference:
        moving_data = moving.read()
        warped = np.zeros((moving.count, reference.height, reference.width), dtype=moving_data.dtype)
        for i in range(moving.count):
            warped[i] = cv2.warpAffine(
                moving_data[i],
                matrix,
                (reference.width, reference.height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        profile = reference.profile.copy()
        profile.update(count=moving.count, dtype=moving_data.dtype, nodata=moving.nodata if moving.nodata is not None else 0)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(warped)


def affine_sanity(matrix: np.ndarray) -> dict:
    linear = matrix[:, :2]
    translation = matrix[:, 2]
    det = float(np.linalg.det(linear))
    scale_x = float(np.linalg.norm(linear[:, 0]))
    scale_y = float(np.linalg.norm(linear[:, 1]))
    trans_norm = float(np.linalg.norm(translation))
    return {
        "determinant": det,
        "scale_x": scale_x,
        "scale_y": scale_y,
        "translation_pixels": trans_norm,
        "near_identity": 0.5 <= scale_x <= 1.5 and 0.5 <= scale_y <= 1.5 and abs(det) > 0.25 and trans_norm <= 100,
    }


def append_log(log_path: str | Path, args: argparse.Namespace, metrics: dict) -> None:
    path = ensure_parent(log_path)
    parts = ["python", "scripts/05_coregister_cfog.py"]
    for key, value in vars(args).items():
        if key == "log":
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                parts.append(flag)
        else:
            parts.extend([flag, str(value)])
    command = " ".join(shlex.quote(part) for part in parts)
    lines = [
        "",
        f"## CFOG-like Run - {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "### Command",
        "",
        "```powershell",
        command,
        "```",
        "",
        "### Metrics",
        "",
        f"- moving: `{metrics['moving']}`",
        f"- reference: `{metrics['reference']}`",
        f"- detected moving/reference: `{metrics['keypoints_moving']}` / `{metrics['keypoints_reference']}`",
        f"- descriptors moving/reference: `{metrics['descriptors_moving']}` / `{metrics['descriptors_reference']}`",
        f"- matches after ratio: `{metrics['matches_after_ratio']}`",
        f"- matches after spatial gate: `{metrics['matches_after_spatial_gate']}`",
        f"- RANSAC inliers: `{metrics['ransac_inliers']}`",
        f"- inlier ratio: `{metrics['inlier_ratio']:.4f}`",
        f"- RMSE: `{metrics['inlier_rmse_pixels']:.4f}` pixels",
        f"- near identity: `{metrics['sanity']['near_identity']}`",
        "",
        "### Assessment",
        "",
        metrics["assessment"],
        "",
    ]
    with path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="CFOG-like SAR-optical registration baseline with structural orientation descriptors.")
    parser.add_argument("--moving", default="data/processed/sar_10m.tif")
    parser.add_argument("--reference", default="data/processed/optical_10m.tif")
    parser.add_argument("--output", default="outputs/sar_warped_to_optical_cfog.tif")
    parser.add_argument("--metrics", default="outputs/metrics_cfog.json")
    parser.add_argument("--matches", default="outputs/matches_cfog.png")
    parser.add_argument("--log", default="EXPERIMENT_LOG.md")
    parser.add_argument("--model", choices=["affine", "partial-affine"], default="partial-affine")
    parser.add_argument("--max-points", type=int, default=3000)
    parser.add_argument("--quality", type=float, default=0.005)
    parser.add_argument("--min-distance", type=float, default=6.0)
    parser.add_argument("--orientations", type=int, default=9)
    parser.add_argument("--patch", type=int, default=32)
    parser.add_argument("--cells", type=int, default=4)
    parser.add_argument("--sigma", type=float, default=1.2)
    parser.add_argument("--ratio", type=float, default=0.95)
    parser.add_argument("--ransac-thresh", type=float, default=5.0)
    parser.add_argument("--max-displacement", type=float, default=80.0, help="Pixel-space gate using the prior geocoded alignment; <=0 disables it.")
    parser.add_argument("--sar-log", action="store_true")
    parser.add_argument("--no-mutual", action="store_true")
    args = parser.parse_args()

    moving_path = Path(args.moving)
    reference_path = Path(args.reference)
    moving_gray, _ = read_first_band_uint8(moving_path, log_scale=args.sar_log)
    reference_gray, _ = read_rgb_as_gray_uint8(reference_path)
    moving_gray = enhance_for_structure(moving_gray)
    reference_gray = enhance_for_structure(reference_gray)

    border = args.patch // 2
    kp_moving = detect_points(moving_gray, args.max_points, args.quality, args.min_distance, border)
    kp_reference = detect_points(reference_gray, args.max_points, args.quality, args.min_distance, border)

    ch_moving = cfog_channels(moving_gray, args.orientations, args.sigma)
    ch_reference = cfog_channels(reference_gray, args.orientations, args.sigma)
    kp_moving, desc_moving = describe_keypoints(ch_moving, kp_moving, args.patch, args.cells)
    kp_reference, desc_reference = describe_keypoints(ch_reference, kp_reference, args.patch, args.cells)
    if len(desc_moving) == 0 or len(desc_reference) == 0:
        raise RuntimeError("No CFOG descriptors were created.")

    matches_ratio = match_descriptors(desc_moving, desc_reference, args.ratio, mutual=not args.no_mutual)
    matches = spatial_gate(matches_ratio, kp_moving, kp_reference, args.max_displacement)
    if len(matches) < 3:
        raise RuntimeError(f"Not enough matches after filtering: {len(matches)}")

    src_pts = np.float32([kp_moving[m.queryIdx].pt for m in matches])
    dst_pts = np.float32([kp_reference[m.trainIdx].pt for m in matches])
    matrix, inliers = estimate_affine(src_pts, dst_pts, args.model, args.ransac_thresh)
    if matrix is None or inliers is None:
        raise RuntimeError("RANSAC transform estimation failed.")

    inlier_count = int(inliers.sum())
    rmse = rmse_affine(src_pts, dst_pts, matrix, inliers)
    sanity = affine_sanity(matrix)
    assessment = (
        "Usable candidate: inlier count is reasonable and affine sanity is close to the geocoded prior."
        if inlier_count >= 10 and sanity["near_identity"]
        else "Not reliable as a final registration transform; keep as an experimental round and inspect match/overlay outputs."
    )

    metrics = {
        "method": "cfog_like",
        "model": args.model,
        "moving": str(moving_path),
        "reference": str(reference_path),
        "keypoints_moving": len(kp_moving),
        "keypoints_reference": len(kp_reference),
        "descriptors_moving": int(desc_moving.shape[0]),
        "descriptors_reference": int(desc_reference.shape[0]),
        "matches_after_ratio": len(matches_ratio),
        "matches_after_spatial_gate": len(matches),
        "ransac_inliers": inlier_count,
        "inlier_ratio": inlier_count / max(len(matches), 1),
        "inlier_rmse_pixels": rmse,
        "transform_matrix": matrix.tolist(),
        "sanity": sanity,
        "assessment": assessment,
        "parameters": {
            "max_points": args.max_points,
            "quality": args.quality,
            "min_distance": args.min_distance,
            "orientations": args.orientations,
            "patch": args.patch,
            "cells": args.cells,
            "sigma": args.sigma,
            "ratio": args.ratio,
            "ransac_thresh": args.ransac_thresh,
            "max_displacement": args.max_displacement,
            "mutual": not args.no_mutual,
            "sar_log": args.sar_log,
        },
    }

    warp_full_resolution(moving_path, reference_path, matrix, ensure_parent(args.output))
    match_vis = cv2.drawMatches(
        moving_gray,
        kp_moving,
        reference_gray,
        kp_reference,
        matches,
        None,
        matchesMask=inliers.ravel().astype(int).tolist(),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imwrite(str(ensure_parent(args.matches)), match_vis)
    write_json(args.metrics, metrics)
    append_log(args.log, args, metrics)

    print(f"Wrote warped image: {args.output}")
    print(f"Wrote matches: {args.matches}")
    print(f"Wrote metrics: {args.metrics}")
    print(f"Appended log: {args.log}")
    print(metrics)


if __name__ == "__main__":
    main()
