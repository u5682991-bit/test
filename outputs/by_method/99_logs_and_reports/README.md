# SAR-Optical Registration Baseline

This project contains a reproducible baseline for registering Sentinel-1 GRD SAR images to Sentinel-2 L2A optical images over the same AOI.

## Directory Layout

```text
data/
  raw/
    sentinel1/
    sentinel2/
  processed/
scripts/
  01_preprocess_sar.py
  02_preprocess_optical.py
  03_coregister_opencv.py
  04_evaluate_overlay.py
outputs/
requirements.txt
```

## Install Python Dependencies

```powershell
python -m pip install -r requirements.txt
```

SNAP Desktop / SNAP GPT is also needed for full Sentinel-1 GRD calibration and terrain correction.

## 1. SAR Preprocessing

Generate and run a SNAP GPT graph:

```powershell
python scripts/01_preprocess_sar.py `
  --input S1C_IW_GRDH_1SDV_20251228T230337_20251228T230402_005658_00B4C9_5CCC.zip `
  --output data/processed/sar_10m.tif `
  --polarizations VV `
  --map-projection EPSG:32648 `
  --pixel-spacing 10 `
  --geo-region "POLYGON((104.02 30.32,104.20 30.32,104.20 30.48,104.02 30.48,104.02 30.32))"
```

If SNAP `gpt` is not on `PATH`, pass its executable path with `--gpt`.

To only create the graph XML:

```powershell
python scripts/01_preprocess_sar.py --input S1C_IW_GRDH_1SDV_20251228T230337_20251228T230402_005658_00B4C9_5CCC.zip --write-graph-only
```

## 2. Optical Preprocessing

Build a Sentinel-2 RGB GeoTIFF:

```powershell
python scripts/02_preprocess_optical.py `
  --input S2B_MSIL2A_20251229T035059_N0511_R104_T48RVU_20251229T060507.SAFE.zip `
  --output data/processed/optical_10m.tif
```

If you already have a SAR grid and want Sentinel-2 resampled onto that grid:

```powershell
python scripts/02_preprocess_optical.py `
  --input S2B_MSIL2A_20251229T035059_N0511_R104_T48RVU_20251229T060507.SAFE.zip `
  --output data/processed/optical_10m.tif `
  --reference data/processed/sar_10m.tif
```

## 3. Baseline Registration

Run SIFT + ratio test + RANSAC affine transform:

```powershell
python scripts/03_coregister_opencv.py `
  --moving data/processed/sar_10m.tif `
  --reference data/processed/optical_10m.tif `
  --method sift `
  --model affine `
  --sar-log
```

Useful alternatives:

```powershell
python scripts/03_coregister_opencv.py --method orb --model affine --sar-log
python scripts/03_coregister_opencv.py --method akaze --model partial-affine --sar-log
python scripts/03_coregister_opencv.py --method sift --model homography --sar-log
```

Outputs:

```text
outputs/sar_warped_to_optical.tif
outputs/matches_sift.png
outputs/metrics.json
```

## 4. Visual Evaluation

```powershell
python scripts/04_evaluate_overlay.py --sar-log
```

Outputs:

```text
outputs/overlay_before.png
outputs/overlay_after.png
outputs/checkerboard_before.png
outputs/checkerboard_after.png
```

## Notes

- Use Sentinel-2 as the reference image and Sentinel-1 as the moving image.
- `EPSG:32648` is a practical UTM choice for Chengdu / Tianfu New Area.
- For the OpenCV baseline, affine is usually more stable than homography on SAR-optical image pairs.
- If SIFT/ORB has too few inliers, try stronger SAR filtering, smaller AOI cropping, or structure descriptors such as HOPC, CFOG, RIFT, or PCSD.
