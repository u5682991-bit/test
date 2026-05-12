from __future__ import annotations

import argparse
import datetime as dt
import shlex
from pathlib import Path

import cv2
import numpy as np
import rasterio

from common import ensure_parent, read_first_band_uint8, read_rgb_as_gray_uint8, write_json


def enhance(image: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(cv2.GaussianBlur(image, (0, 0), 0.8))


def rift_maps(image: np.ndarray, orientations: int, wavelengths: list[float], gamma: float) -> tuple[np.ndarray, np.ndarray]:
    img = image.astype(np.float32) / 255.0
    responses = []
    for i in range(orientations):
        theta = i * np.pi / orientations
        oriented = np.zeros_like(img, dtype=np.float32)
        for lambd in wavelengths:
            ksize = int(max(9, round(lambd * 4)))
            if ksize % 2 == 0:
                ksize += 1
            kernel = cv2.getGaborKernel((ksize, ksize), sigma=0.56 * lambd, theta=theta, lambd=lambd, gamma=gamma, psi=0, ktype=cv2.CV_32F)
            kernel -= kernel.mean()
            norm = np.linalg.norm(kernel)
            if norm > 1e-6:
                kernel /= norm
            real = cv2.filter2D(img, cv2.CV_32F, kernel)
            oriented = np.maximum(oriented, np.abs(real))
        responses.append(oriented)
    stack = np.stack(responses, axis=-1)
    max_response = stack.max(axis=-1)
    mim = stack.argmax(axis=-1).astype(np.uint8)
    max_response = cv2.GaussianBlur(max_response, (0, 0), 1.0)
    return max_response, mim


def detect_points(response: np.ndarray, max_points: int, quality: float, min_distance: float, border: int) -> list[cv2.KeyPoint]:
    img = cv2.normalize(response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    pts = cv2.goodFeaturesToTrack(
        img,
        maxCorners=max_points,
        qualityLevel=quality,
        minDistance=min_distance,
        blockSize=5,
        useHarrisDetector=True,
        k=0.04,
    )
    if pts is None:
        return []
    h, w = img.shape
    out = []
    for x, y in pts.reshape(-1, 2):
        if border <= x < w - border and border <= y < h - border:
            out.append(cv2.KeyPoint(float(x), float(y), float(border)))
    return out


def describe_mim(
    mim: np.ndarray,
    response: np.ndarray,
    keypoints: list[cv2.KeyPoint],
    orientations: int,
    patch: int,
    cells: int,
) -> tuple[list[cv2.KeyPoint], np.ndarray]:
    radius = patch // 2
    cell = patch // cells
    h, w = mim.shape
    valid = []
    descs = []
    bins = np.arange(orientations + 1)

    for kp in keypoints:
        x = int(round(kp.pt[0]))
        y = int(round(kp.pt[1]))
        if x - radius < 0 or y - radius < 0 or x + radius > w or y + radius > h:
            continue
        mim_roi = mim[y - radius : y + radius, x - radius : x + radius]
        rsp_roi = response[y - radius : y + radius, x - radius : x + radius]
        feat = []
        for yy in range(cells):
            for xx in range(cells):
                labels = mim_roi[yy * cell : (yy + 1) * cell, xx * cell : (xx + 1) * cell].ravel()
                weights = rsp_roi[yy * cell : (yy + 1) * cell, xx * cell : (xx + 1) * cell].ravel()
                hist, _ = np.histogram(labels, bins=bins, weights=weights)
                feat.extend(hist.tolist())
        desc = np.asarray(feat, dtype=np.float32)
        norm = np.linalg.norm(desc)
        if norm > 1e-6:
            desc /= norm
            valid.append(kp)
            descs.append(desc)

    if not descs:
        return [], np.empty((0, cells * cells * orientations), dtype=np.float32)
    return valid, np.vstack(descs).astype(np.float32)


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
    out = []
    for m in matches:
        x1, y1 = kp_moving[m.queryIdx].pt
        x2, y2 = kp_reference[m.trainIdx].pt
        if np.hypot(x2 - x1, y2 - y1) <= max_displacement:
            out.append(m)
    return out


def estimate(src_pts: np.ndarray, dst_pts: np.ndarray, model: str, thresh: float):
    if model == "partial-affine":
        return cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=thresh)
    if model == "affine":
        return cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=thresh)
    raise ValueError(model)


def rmse(src_pts: np.ndarray, dst_pts: np.ndarray, matrix: np.ndarray, inliers: np.ndarray) -> float:
    mask = inliers.ravel().astype(bool)
    pred = cv2.transform(src_pts[mask].reshape(-1, 1, 2), matrix).reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum((pred - dst_pts[mask]) ** 2, axis=1)))) if len(pred) else float("nan")


def sanity(matrix: np.ndarray) -> dict:
    linear = matrix[:, :2]
    translation = matrix[:, 2]
    det = float(np.linalg.det(linear))
    scale_x = float(np.linalg.norm(linear[:, 0]))
    scale_y = float(np.linalg.norm(linear[:, 1]))
    trans = float(np.linalg.norm(translation))
    return {
        "determinant": det,
        "scale_x": scale_x,
        "scale_y": scale_y,
        "translation_pixels": trans,
        "near_identity": 0.5 <= scale_x <= 1.5 and 0.5 <= scale_y <= 1.5 and abs(det) > 0.25 and trans <= 100,
    }


def warp(moving_path: Path, reference_path: Path, matrix: np.ndarray, output_path: Path) -> None:
    with rasterio.open(moving_path) as moving, rasterio.open(reference_path) as reference:
        data = moving.read()
        out = np.zeros((moving.count, reference.height, reference.width), dtype=data.dtype)
        for i in range(moving.count):
            out[i] = cv2.warpAffine(data[i], matrix, (reference.width, reference.height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        profile = reference.profile.copy()
        profile.update(count=moving.count, dtype=data.dtype, nodata=moving.nodata if moving.nodata is not None else 0)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(out)


def append_log(path: str | Path, args: argparse.Namespace, metrics: dict) -> None:
    parts = ["python", "scripts/06_coregister_rift.py"]
    for key, value in vars(args).items():
        if key == "log":
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                parts.append(flag)
        else:
            parts.extend([flag, str(value)])
    command = " ".join(shlex.quote(p) for p in parts)
    lines = [
        "",
        f"## RIFT-like Run - {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "### Command",
        "",
        "```powershell",
        command,
        "```",
        "",
        "### Metrics",
        "",
        f"- detected moving/reference: `{metrics['keypoints_moving']}` / `{metrics['keypoints_reference']}`",
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
    with ensure_parent(path).open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="RIFT-like SAR-optical registration using MIM orientation histograms.")
    parser.add_argument("--moving", default="data/processed/sar_10m.tif")
    parser.add_argument("--reference", default="data/processed/optical_10m.tif")
    parser.add_argument("--output", default="outputs/sar_warped_to_optical_rift.tif")
    parser.add_argument("--metrics", default="outputs/metrics_rift.json")
    parser.add_argument("--matches", default="outputs/matches_rift.png")
    parser.add_argument("--log", default="EXPERIMENT_LOG.md")
    parser.add_argument("--model", choices=["partial-affine", "affine"], default="partial-affine")
    parser.add_argument("--max-points", type=int, default=5000)
    parser.add_argument("--quality", type=float, default=0.003)
    parser.add_argument("--min-distance", type=float, default=5.0)
    parser.add_argument("--orientations", type=int, default=12)
    parser.add_argument("--wavelengths", default="4,8,16")
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--patch", type=int, default=48)
    parser.add_argument("--cells", type=int, default=4)
    parser.add_argument("--ratio", type=float, default=0.98)
    parser.add_argument("--ransac-thresh", type=float, default=4.0)
    parser.add_argument("--max-displacement", type=float, default=50.0)
    parser.add_argument("--sar-log", action="store_true")
    parser.add_argument("--no-mutual", action="store_true")
    args = parser.parse_args()

    wavelengths = [float(x) for x in args.wavelengths.split(",") if x.strip()]
    moving_path = Path(args.moving)
    reference_path = Path(args.reference)
    moving, _ = read_first_band_uint8(moving_path, log_scale=args.sar_log)
    reference, _ = read_rgb_as_gray_uint8(reference_path)
    moving = enhance(moving)
    reference = enhance(reference)

    rsp_moving, mim_moving = rift_maps(moving, args.orientations, wavelengths, args.gamma)
    rsp_reference, mim_reference = rift_maps(reference, args.orientations, wavelengths, args.gamma)
    border = args.patch // 2
    kp_moving = detect_points(rsp_moving, args.max_points, args.quality, args.min_distance, border)
    kp_reference = detect_points(rsp_reference, args.max_points, args.quality, args.min_distance, border)
    kp_moving, desc_moving = describe_mim(mim_moving, rsp_moving, kp_moving, args.orientations, args.patch, args.cells)
    kp_reference, desc_reference = describe_mim(mim_reference, rsp_reference, kp_reference, args.orientations, args.patch, args.cells)
    if len(desc_moving) == 0 or len(desc_reference) == 0:
        raise RuntimeError("No RIFT-like descriptors were created.")

    matches_ratio = match_descriptors(desc_moving, desc_reference, args.ratio, mutual=not args.no_mutual)
    matches = spatial_gate(matches_ratio, kp_moving, kp_reference, args.max_displacement)
    if len(matches) < 3:
        raise RuntimeError(f"Not enough matches after filtering: {len(matches)}")
    src_pts = np.float32([kp_moving[m.queryIdx].pt for m in matches])
    dst_pts = np.float32([kp_reference[m.trainIdx].pt for m in matches])
    matrix, inliers = estimate(src_pts, dst_pts, args.model, args.ransac_thresh)
    if matrix is None or inliers is None:
        raise RuntimeError("RANSAC transform estimation failed.")

    inlier_count = int(inliers.sum())
    matrix_sanity = sanity(matrix)
    metrics = {
        "method": "rift_like_mim",
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
        "inlier_rmse_pixels": rmse(src_pts, dst_pts, matrix, inliers),
        "transform_matrix": matrix.tolist(),
        "sanity": matrix_sanity,
        "parameters": {
            "max_points": args.max_points,
            "quality": args.quality,
            "min_distance": args.min_distance,
            "orientations": args.orientations,
            "wavelengths": wavelengths,
            "gamma": args.gamma,
            "patch": args.patch,
            "cells": args.cells,
            "ratio": args.ratio,
            "ransac_thresh": args.ransac_thresh,
            "max_displacement": args.max_displacement,
            "mutual": not args.no_mutual,
            "sar_log": args.sar_log,
        },
    }
    metrics["assessment"] = (
        "Usable RIFT-like candidate: inlier count and affine sanity are acceptable."
        if inlier_count >= 10 and matrix_sanity["near_identity"]
        else "Not reliable as a final registration transform; keep as an experimental round."
    )

    warp(moving_path, reference_path, matrix, ensure_parent(args.output))
    match_vis = cv2.drawMatches(
        moving,
        kp_moving,
        reference,
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
