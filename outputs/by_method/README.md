# Method-Based Output Archive

This directory groups the SAR-Optical registration outputs by method. Files were copied here for reporting and inspection; the original outputs remain in their original locations.

## Folders

```text
00_data_and_preprocessing/
```

Input-grid and preprocessing outputs:

- `sar_10m.tif`
- `optical_10m.tif`
- `data_quality.json`
- SNAP graph XML

```text
01_opencv_baseline_sift_orb_akaze/
```

Traditional OpenCV baselines:

- SIFT affine / homography
- ORB partial-affine
- AKAZE partial-affine

```text
02_cfog_like/
```

CFOG-like structural orientation descriptor experiments.

```text
03_rift_like/
```

RIFT-like MIM descriptor experiments.

```text
04_sar_filtering_plus_rift/
```

SAR filtering variants and RIFT-like comparison:

- Gaussian
- Median
- Bilateral
- Lee
- Anisotropic diffusion

```text
05_hopc_like/
```

HOPC-like phase-congruency orientation histogram experiments.

```text
06_pcsd_like/
```

PCSD-like phase-congruency structural descriptor experiments.

```text
07_hopc_optimized/
```

Focused HOPC-like optimization results.

Important candidates:

- `metrics_hopc_opt_r2.json`: best RMSE with at least 20 inliers.
- `metrics_hopc_opt_r8.json`: balanced candidate with at least 35 inliers.
- `hopc_opt_summary.json`: summary table.

```text
98_scripts/
```

All scripts used to generate the experiments.

```text
99_logs_and_reports/
```

Experiment logs and Markdown reports.

## Current Best Results

Best precision candidate:

```text
HOPC opt R2
inliers: 27
RMSE: 1.26 px
```

Best balanced candidate:

```text
HOPC opt R8
inliers: 37
RMSE: 1.78 px
```

Best earlier high-inlier HOPC candidate:

```text
HOPC R1
inliers: 71
RMSE: 2.63 px
```

