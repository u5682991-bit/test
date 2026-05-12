from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from common import ensure_parent, read_first_band_uint8, read_rgb_as_gray_uint8


def save_overlay(sar_gray: np.ndarray, optical_gray: np.ndarray, path: str | Path) -> None:
    sar_edges = cv2.Canny(sar_gray, 60, 140)
    optical_rgb = cv2.cvtColor(optical_gray, cv2.COLOR_GRAY2RGB)
    optical_rgb[sar_edges > 0] = [255, 0, 0]
    cv2.imwrite(str(ensure_parent(path)), cv2.cvtColor(optical_rgb, cv2.COLOR_RGB2BGR))


def save_checkerboard(a: np.ndarray, b: np.ndarray, path: str | Path, tile: int = 96) -> None:
    h, w = a.shape
    yy, xx = np.indices((h, w))
    mask = ((xx // tile + yy // tile) % 2).astype(bool)
    out = np.where(mask, a, b).astype(np.uint8)
    cv2.imwrite(str(ensure_parent(path)), out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create visual overlays for SAR-optical registration inspection.")
    parser.add_argument("--sar-before", default="data/processed/sar_10m.tif")
    parser.add_argument("--sar-after", default="outputs/sar_warped_to_optical.tif")
    parser.add_argument("--optical", default="data/processed/optical_10m.tif")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--sar-log", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    optical_gray, _ = read_rgb_as_gray_uint8(args.optical)
    sar_before, _ = read_first_band_uint8(args.sar_before, log_scale=args.sar_log)
    sar_after, _ = read_first_band_uint8(args.sar_after, log_scale=args.sar_log)

    h = min(optical_gray.shape[0], sar_before.shape[0], sar_after.shape[0])
    w = min(optical_gray.shape[1], sar_before.shape[1], sar_after.shape[1])
    optical_gray = optical_gray[:h, :w]
    sar_before = sar_before[:h, :w]
    sar_after = sar_after[:h, :w]

    save_overlay(sar_before, optical_gray, out_dir / "overlay_before.png")
    save_overlay(sar_after, optical_gray, out_dir / "overlay_after.png")
    save_checkerboard(sar_before, optical_gray, out_dir / "checkerboard_before.png")
    save_checkerboard(sar_after, optical_gray, out_dir / "checkerboard_after.png")

    print(f"Wrote overlays to: {out_dir}")


if __name__ == "__main__":
    main()
