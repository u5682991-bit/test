from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import from_bounds


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, data: dict) -> None:
    path = ensure_parent(path)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def percentile_uint8(image: np.ndarray, nodata: Optional[float] = None) -> np.ndarray:
    arr = image.astype(np.float32, copy=False)
    mask = np.isfinite(arr)
    if nodata is not None:
        mask &= arr != nodata
    if not np.any(mask):
        return np.zeros(arr.shape, dtype=np.uint8)

    valid = arr[mask]
    lo, hi = np.percentile(valid, (2, 98))
    if hi <= lo:
        hi = float(valid.max())
        lo = float(valid.min())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)

    out = np.clip((arr - lo) / (hi - lo), 0, 1)
    out[~mask] = 0
    return (out * 255).astype(np.uint8)


def read_first_band_uint8(path: str | Path, log_scale: bool = False) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        band = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        nodata = src.nodata
    if log_scale:
        band = np.where(np.isfinite(band), band, 0)
        band = np.log1p(np.maximum(band, 0))
    return percentile_uint8(band, nodata), profile


def read_rgb_as_gray_uint8(path: str | Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        profile = src.profile.copy()
        if src.count >= 3:
            rgb = src.read([1, 2, 3]).transpose(1, 2, 0)
            rgb8 = np.dstack([percentile_uint8(rgb[:, :, i], src.nodata) for i in range(3)])
            gray = cv2.cvtColor(rgb8, cv2.COLOR_RGB2GRAY)
        else:
            gray = percentile_uint8(src.read(1), src.nodata)
    return gray, profile


def reproject_to_match(
    source_path: str | Path,
    reference_path: str | Path,
    output_path: str | Path,
    resampling: Resampling = Resampling.bilinear,
) -> Path:
    output_path = ensure_parent(output_path)
    with rasterio.open(reference_path) as ref, rasterio.open(source_path) as src:
        profile = ref.profile.copy()
        profile.update(count=src.count, dtype=src.dtypes[0], nodata=src.nodata)
        with WarpedVRT(
            src,
            crs=ref.crs,
            transform=ref.transform,
            width=ref.width,
            height=ref.height,
            resampling=resampling,
        ) as vrt:
            data = vrt.read()
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data)
    return output_path


def crop_to_bounds(
    input_path: str | Path,
    output_path: str | Path,
    bounds: Tuple[float, float, float, float],
) -> Path:
    output_path = ensure_parent(output_path)
    with rasterio.open(input_path) as src:
        window = from_bounds(*bounds, transform=src.transform)
        data = src.read(window=window)
        transform = src.window_transform(window)
        profile = src.profile.copy()
        profile.update(height=data.shape[1], width=data.shape[2], transform=transform)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data)
    return output_path


def find_one(paths: Iterable[Path], patterns: Iterable[str]) -> Path:
    for pattern in patterns:
        matches = sorted(p for root in paths for p in root.rglob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No file matched any pattern: {', '.join(patterns)}")
