from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import rasterio

from common import ensure_parent, read_first_band_uint8, read_rgb_as_gray_uint8, write_json


def make_detector(name: str, nfeatures: int):
    method = name.lower()
    if method == "sift":
        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError("This OpenCV build does not include SIFT. Try --method orb or install opencv-contrib-python.")
        return cv2.SIFT_create(nfeatures=nfeatures), cv2.NORM_L2
    if method == "orb":
        return cv2.ORB_create(nfeatures=nfeatures), cv2.NORM_HAMMING
    if method == "akaze":
        return cv2.AKAZE_create(), cv2.NORM_HAMMING
    raise ValueError(f"Unsupported method: {name}")


def match_descriptors(desc_moving, desc_reference, norm_type: int, ratio: float):
    if norm_type == cv2.NORM_L2:
        matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
        d1 = desc_moving.astype(np.float32)
        d2 = desc_reference.astype(np.float32)
    else:
        matcher = cv2.BFMatcher(norm_type)
        d1, d2 = desc_moving, desc_reference

    pairs = matcher.knnMatch(d1, d2, k=2)
    good = []
    for pair in pairs:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio * n.distance:
                good.append(m)
    return good


def estimate_transform(src_pts: np.ndarray, dst_pts: np.ndarray, model: str, ransac_thresh: float):
    if model == "affine":
        matrix, inliers = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh)
        return matrix, inliers, "affine"
    if model == "partial-affine":
        matrix, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh)
        return matrix, inliers, "affine"
    if model == "homography":
        matrix, inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
        return matrix, inliers, "homography"
    raise ValueError(f"Unsupported transform model: {model}")


def reprojection_rmse(src_pts: np.ndarray, dst_pts: np.ndarray, matrix: np.ndarray, transform_kind: str, inlier_mask: np.ndarray) -> float:
    mask = inlier_mask.ravel().astype(bool)
    src = src_pts[mask]
    dst = dst_pts[mask]
    if len(src) == 0:
        return float("nan")
    if transform_kind == "affine":
        pred = cv2.transform(src.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    else:
        pred = cv2.perspectiveTransform(src.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum((pred - dst) ** 2, axis=1))))


def warp_full_resolution(moving_path: Path, reference_path: Path, matrix: np.ndarray, transform_kind: str, output_path: Path) -> None:
    with rasterio.open(moving_path) as moving, rasterio.open(reference_path) as reference:
        moving_data = moving.read()
        warped = np.zeros((moving.count, reference.height, reference.width), dtype=moving_data.dtype)
        for i in range(moving.count):
            if transform_kind == "affine":
                warped[i] = cv2.warpAffine(
                    moving_data[i],
                    matrix,
                    (reference.width, reference.height),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
            else:
                warped[i] = cv2.warpPerspective(
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


def main() -> None:
    parser = argparse.ArgumentParser(description="SAR-optical image registration baseline with OpenCV features and RANSAC.")
    parser.add_argument("--moving", default="data/processed/sar_10m.tif", help="Moving image, usually SAR")
    parser.add_argument("--reference", default="data/processed/optical_10m.tif", help="Reference image, usually optical")
    parser.add_argument("--output", default="outputs/sar_warped_to_optical.tif")
    parser.add_argument("--metrics", default="outputs/metrics.json")
    parser.add_argument("--matches", default="outputs/matches_sift.png")
    parser.add_argument("--method", choices=["sift", "orb", "akaze"], default="sift")
    parser.add_argument("--model", choices=["affine", "partial-affine", "homography"], default="affine")
    parser.add_argument("--nfeatures", type=int, default=8000)
    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--ransac-thresh", type=float, default=5.0)
    parser.add_argument("--sar-log", action="store_true", help="Apply log1p scaling to the moving image for feature extraction")
    args = parser.parse_args()

    moving_path = Path(args.moving)
    reference_path = Path(args.reference)
    moving_gray, _ = read_first_band_uint8(moving_path, log_scale=args.sar_log)
    reference_gray, _ = read_rgb_as_gray_uint8(reference_path)

    detector, norm_type = make_detector(args.method, args.nfeatures)
    kp_moving, desc_moving = detector.detectAndCompute(moving_gray, None)
    kp_reference, desc_reference = detector.detectAndCompute(reference_gray, None)
    if desc_moving is None or desc_reference is None:
        raise RuntimeError("Feature extraction failed on one of the images.")

    matches = match_descriptors(desc_moving, desc_reference, norm_type, args.ratio)
    min_matches = 4 if args.model == "homography" else 3
    if len(matches) < min_matches:
        raise RuntimeError(f"Not enough matches: got {len(matches)}, need at least {min_matches}.")

    src_pts = np.float32([kp_moving[m.queryIdx].pt for m in matches])
    dst_pts = np.float32([kp_reference[m.trainIdx].pt for m in matches])
    matrix, inliers, transform_kind = estimate_transform(src_pts, dst_pts, args.model, args.ransac_thresh)
    if matrix is None or inliers is None:
        raise RuntimeError("RANSAC transform estimation failed.")

    rmse = reprojection_rmse(src_pts, dst_pts, matrix, transform_kind, inliers)
    inlier_count = int(inliers.sum())
    metrics = {
        "method": args.method,
        "model": args.model,
        "moving": str(moving_path),
        "reference": str(reference_path),
        "keypoints_moving": len(kp_moving),
        "keypoints_reference": len(kp_reference),
        "matches_after_ratio": len(matches),
        "ransac_inliers": inlier_count,
        "inlier_ratio": inlier_count / max(len(matches), 1),
        "inlier_rmse_pixels": rmse,
        "transform_matrix": matrix.tolist(),
    }

    output_path = ensure_parent(args.output)
    warp_full_resolution(moving_path, reference_path, matrix, transform_kind, output_path)

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

    print(f"Wrote warped image: {output_path}")
    print(f"Wrote matches: {args.matches}")
    print(f"Wrote metrics: {args.metrics}")
    print(metrics)


if __name__ == "__main__":
    main()
