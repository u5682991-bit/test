from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from common import ensure_parent, find_one


def extract_safe_if_needed(input_path: Path, work_dir: Path) -> Path:
    if input_path.suffix.lower() != ".zip":
        return input_path
    safe_dir = work_dir / input_path.stem
    if safe_dir.exists():
        return safe_dir
    safe_dir.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(input_path) as zf:
        zf.extractall(work_dir)
    candidates = sorted(work_dir.glob("*.SAFE"))
    if not candidates:
        raise FileNotFoundError("No .SAFE directory found after extracting Sentinel-2 zip")
    return candidates[0]


def read_band_to_reference(path: Path, ref_profile: dict, ref_crs, ref_transform, ref_width: int, ref_height: int) -> np.ndarray:
    with rasterio.open(path) as src:
        with WarpedVRT(
            src,
            crs=ref_crs,
            transform=ref_transform,
            width=ref_width,
            height=ref_height,
            resampling=Resampling.bilinear,
        ) as vrt:
            return vrt.read(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Sentinel-2 L2A RGB GeoTIFF from B04/B03/B02 bands.")
    parser.add_argument("--input", required=True, help="Sentinel-2 L2A .SAFE directory or .SAFE.zip")
    parser.add_argument("--output", default="data/processed/optical_10m.tif")
    parser.add_argument("--work-dir", default="data/raw/sentinel2")
    parser.add_argument("--reference", help="Optional raster whose grid should be matched")
    args = parser.parse_args()

    input_path = Path(args.input)
    safe_dir = extract_safe_if_needed(input_path, Path(args.work_dir))
    roots = [safe_dir]
    b04 = find_one(roots, ["*B04_10m.jp2", "*_B04.jp2"])
    b03 = find_one(roots, ["*B03_10m.jp2", "*_B03.jp2"])
    b02 = find_one(roots, ["*B02_10m.jp2", "*_B02.jp2"])

    if args.reference:
        with rasterio.open(args.reference) as ref:
            profile = ref.profile.copy()
            profile.update(count=3, dtype="uint16", nodata=0)
            bands = [
                read_band_to_reference(b04, profile, ref.crs, ref.transform, ref.width, ref.height),
                read_band_to_reference(b03, profile, ref.crs, ref.transform, ref.width, ref.height),
                read_band_to_reference(b02, profile, ref.crs, ref.transform, ref.width, ref.height),
            ]
    else:
        with rasterio.open(b04) as ref:
            profile = ref.profile.copy()
            profile.update(count=3, dtype="uint16", nodata=0)
            bands = [ref.read(1), read_band_to_reference(b03, profile, ref.crs, ref.transform, ref.width, ref.height), read_band_to_reference(b02, profile, ref.crs, ref.transform, ref.width, ref.height)]

    output = ensure_parent(args.output)
    with rasterio.open(output, "w", **profile) as dst:
        dst.write(np.stack(bands).astype(np.uint16))

    print(f"Wrote optical RGB GeoTIFF: {output}")


if __name__ == "__main__":
    main()
