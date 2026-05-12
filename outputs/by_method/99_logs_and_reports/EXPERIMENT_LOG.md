# Experiment Log

This file records each registration experiment round, including parameters, metrics, outputs, and assessment.


## CFOG-like Run - 2026-05-12 20:14:31

### Command

```powershell
python scripts/05_coregister_cfog.py --moving 'data\processed\sar_10m.tif' --reference 'data\processed\optical_10m.tif' --output 'outputs\sar_warped_to_optical_cfog_r1.tif' --metrics 'outputs\metrics_cfog_r1.json' --matches 'outputs\matches_cfog_r1.png' --model partial-affine --max-points 3000 --quality 0.005 --min-distance 6.0 --orientations 9 --patch 32 --cells 4 --sigma 1.2 --ratio 0.95 --ransac-thresh 5.0 --max-displacement 80.0 --sar-log
```

### Metrics

- moving: `data\processed\sar_10m.tif`
- reference: `data\processed\optical_10m.tif`
- detected moving/reference: `2631` / `2266`
- descriptors moving/reference: `2631` / `2266`
- matches after ratio: `160`
- matches after spatial gate: `14`
- RANSAC inliers: `5`
- inlier ratio: `0.3571`
- RMSE: `1.4087` pixels
- near identity: `True`

### Assessment

Not reliable as a final registration transform; keep as an experimental round and inspect match/overlay outputs.

## CFOG-like Run - 2026-05-12 20:15:27

### Command

```powershell
python scripts/05_coregister_cfog.py --moving 'data\processed\sar_10m.tif' --reference 'data\processed\optical_10m.tif' --output 'outputs\sar_warped_to_optical_cfog_r2.tif' --metrics 'outputs\metrics_cfog_r2.json' --matches 'outputs\matches_cfog_r2.png' --model partial-affine --max-points 5000 --quality 0.003 --min-distance 5.0 --orientations 9 --patch 32 --cells 4 --sigma 1.2 --ratio 0.98 --ransac-thresh 5.0 --max-displacement 50.0 --sar-log --no-mutual
```

### Metrics

- moving: `data\processed\sar_10m.tif`
- reference: `data\processed\optical_10m.tif`
- detected moving/reference: `4445` / `3189`
- descriptors moving/reference: `4445` / `3189`
- matches after ratio: `2804`
- matches after spatial gate: `82`
- RANSAC inliers: `11`
- inlier ratio: `0.1341`
- RMSE: `2.7335` pixels
- near identity: `True`

### Assessment

Usable candidate: inlier count is reasonable and affine sanity is close to the geocoded prior.

## CFOG-like Run - 2026-05-12 20:15:29

### Command

```powershell
python scripts/05_coregister_cfog.py --moving 'data\processed\sar_10m.tif' --reference 'data\processed\optical_10m.tif' --output 'outputs\sar_warped_to_optical_cfog_r3.tif' --metrics 'outputs\metrics_cfog_r3.json' --matches 'outputs\matches_cfog_r3.png' --model partial-affine --max-points 5000 --quality 0.003 --min-distance 5.0 --orientations 12 --patch 40 --cells 4 --sigma 1.5 --ratio 0.98 --ransac-thresh 5.0 --max-displacement 50.0 --sar-log --no-mutual
```

### Metrics

- moving: `data\processed\sar_10m.tif`
- reference: `data\processed\optical_10m.tif`
- detected moving/reference: `4312` / `3093`
- descriptors moving/reference: `4312` / `3093`
- matches after ratio: `2685`
- matches after spatial gate: `94`
- RANSAC inliers: `15`
- inlier ratio: `0.1596`
- RMSE: `2.7745` pixels
- near identity: `True`

### Assessment

Usable candidate: inlier count is reasonable and affine sanity is close to the geocoded prior.

## CFOG-like Run - 2026-05-12 20:16:37

### Command

```powershell
python scripts/05_coregister_cfog.py --moving 'data\processed\sar_10m.tif' --reference 'data\processed\optical_10m.tif' --output 'outputs\sar_warped_to_optical_cfog_r4.tif' --metrics 'outputs\metrics_cfog_r4.json' --matches 'outputs\matches_cfog_r4.png' --model partial-affine --max-points 5000 --quality 0.003 --min-distance 5.0 --orientations 12 --patch 40 --cells 4 --sigma 1.5 --ratio 0.98 --ransac-thresh 3.0 --max-displacement 30.0 --sar-log --no-mutual
```

### Metrics

- moving: `data\processed\sar_10m.tif`
- reference: `data\processed\optical_10m.tif`
- detected moving/reference: `4312` / `3093`
- descriptors moving/reference: `4312` / `3093`
- matches after ratio: `2685`
- matches after spatial gate: `55`
- RANSAC inliers: `9`
- inlier ratio: `0.1636`
- RMSE: `1.3929` pixels
- near identity: `True`

### Assessment

Not reliable as a final registration transform; keep as an experimental round and inspect match/overlay outputs.

## CFOG-like Run - 2026-05-12 20:16:47

### Command

```powershell
python scripts/05_coregister_cfog.py --moving 'data\processed\sar_10m.tif' --reference 'data\processed\optical_10m.tif' --output 'outputs\sar_warped_to_optical_cfog_r5.tif' --metrics 'outputs\metrics_cfog_r5.json' --matches 'outputs\matches_cfog_r5.png' --model partial-affine --max-points 7000 --quality 0.002 --min-distance 4.0 --orientations 12 --patch 48 --cells 4 --sigma 1.8 --ratio 0.98 --ransac-thresh 4.0 --max-displacement 50.0 --sar-log --no-mutual
```

### Metrics

- moving: `data\processed\sar_10m.tif`
- reference: `data\processed\optical_10m.tif`
- detected moving/reference: `5626` / `4057`
- descriptors moving/reference: `5626` / `4057`
- matches after ratio: `3369`
- matches after spatial gate: `113`
- RANSAC inliers: `20`
- inlier ratio: `0.1770`
- RMSE: `2.4945` pixels
- near identity: `True`

### Assessment

Usable candidate: inlier count is reasonable and affine sanity is close to the geocoded prior.

## Visual Evaluation - CFOG-like R5

Generated overlay and checkerboard outputs for the current best CFOG-like candidate.

- overlay before: `outputs/cfog_r5/overlay_before.png`
- overlay after: `outputs/cfog_r5/overlay_after.png`
- checkerboard before: `outputs/cfog_r5/checkerboard_before.png`
- checkerboard after: `outputs/cfog_r5/checkerboard_after.png`

Assessment: R5 is the current best automatic candidate by inlier count, but it should still be visually inspected because SAR-Optical local features remain sparse.


## RIFT-like Run - 2026-05-12 20:22:16

### Command

```powershell
python scripts/06_coregister_rift.py --moving 'data\processed\sar_10m.tif' --reference 'data\processed\optical_10m.tif' --output 'outputs\sar_warped_to_optical_rift_r1.tif' --metrics 'outputs\metrics_rift_r1.json' --matches 'outputs\matches_rift_r1.png' --model partial-affine --max-points 5000 --quality 0.003 --min-distance 5.0 --orientations 12 --wavelengths 4,8,16 --gamma 0.5 --patch 48 --cells 4 --ratio 0.98 --ransac-thresh 4.0 --max-displacement 50.0 --sar-log --no-mutual
```

### Metrics

- detected moving/reference: `3131` / `2339`
- matches after ratio: `1660`
- matches after spatial gate: `194`
- RANSAC inliers: `27`
- inlier ratio: `0.1392`
- RMSE: `2.5332` pixels
- near identity: `True`

### Assessment

Usable RIFT-like candidate: inlier count and affine sanity are acceptable.

## RIFT-like Run - 2026-05-12 20:22:43

### Command

```powershell
python scripts/06_coregister_rift.py --moving 'data\processed\sar_10m.tif' --reference 'data\processed\optical_10m.tif' --output 'outputs\sar_warped_to_optical_rift_r4.tif' --metrics 'outputs\metrics_rift_r4.json' --matches 'outputs\matches_rift_r4.png' --model partial-affine --max-points 7000 --quality 0.002 --min-distance 4.0 --orientations 12 --wavelengths 6,12,24 --gamma 0.5 --patch 56 --cells 4 --ratio 0.98 --ransac-thresh 4.0 --max-displacement 50.0 --sar-log --no-mutual
```

### Metrics

- detected moving/reference: `2777` / `2669`
- matches after ratio: `1218`
- matches after spatial gate: `206`
- RANSAC inliers: `23`
- inlier ratio: `0.1117`
- RMSE: `2.2885` pixels
- near identity: `True`

### Assessment

Usable RIFT-like candidate: inlier count and affine sanity are acceptable.

## RIFT-like Run - 2026-05-12 20:22:43

### Command

```powershell
python scripts/06_coregister_rift.py --moving 'data\processed\sar_10m.tif' --reference 'data\processed\optical_10m.tif' --output 'outputs\sar_warped_to_optical_rift_r2.tif' --metrics 'outputs\metrics_rift_r2.json' --matches 'outputs\matches_rift_r2.png' --model partial-affine --max-points 7000 --quality 0.002 --min-distance 4.0 --orientations 12 --wavelengths 4,8,16,24 --gamma 0.5 --patch 48 --cells 4 --ratio 0.98 --ransac-thresh 4.0 --max-displacement 50.0 --sar-log --no-mutual
```

### Metrics

- detected moving/reference: `2886` / `2716`
- matches after ratio: `1478`
- matches after spatial gate: `221`
- RANSAC inliers: `21`
- inlier ratio: `0.0950`
- RMSE: `2.4497` pixels
- near identity: `True`

### Assessment

Usable RIFT-like candidate: inlier count and affine sanity are acceptable.

## RIFT-like Run - 2026-05-12 20:22:43

### Command

```powershell
python scripts/06_coregister_rift.py --moving 'data\processed\sar_10m.tif' --reference 'data\processed\optical_10m.tif' --output 'outputs\sar_warped_to_optical_rift_r3.tif' --metrics 'outputs\metrics_rift_r3.json' --matches 'outputs\matches_rift_r3.png' --model partial-affine --max-points 7000 --quality 0.002 --min-distance 4.0 --orientations 16 --wavelengths 4,8,16 --gamma 0.5 --patch 48 --cells 4 --ratio 0.98 --ransac-thresh 4.0 --max-displacement 50.0 --sar-log --no-mutual
```

### Metrics

- detected moving/reference: `3490` / `2943`
- matches after ratio: `1598`
- matches after spatial gate: `265`
- RANSAC inliers: `42`
- inlier ratio: `0.1585`
- RMSE: `2.5178` pixels
- near identity: `True`

### Assessment

Usable RIFT-like candidate: inlier count and affine sanity are acceptable.

## RIFT-like Run - 2026-05-12 20:23:05

### Command

```powershell
python scripts/06_coregister_rift.py --moving 'data\processed\sar_10m.tif' --reference 'data\processed\optical_10m.tif' --output 'outputs\sar_warped_to_optical_rift_r6.tif' --metrics 'outputs\metrics_rift_r6.json' --matches 'outputs\matches_rift_r6.png' --model partial-affine --max-points 7000 --quality 0.002 --min-distance 4.0 --orientations 16 --wavelengths 4,8,16 --gamma 0.5 --patch 48 --cells 4 --ratio 0.99 --ransac-thresh 4.0 --max-displacement 50.0 --sar-log
```

### Metrics

- detected moving/reference: `3490` / `2943`
- matches after ratio: `331`
- matches after spatial gate: `67`
- RANSAC inliers: `11`
- inlier ratio: `0.1642`
- RMSE: `2.1717` pixels
- near identity: `True`

### Assessment

Usable RIFT-like candidate: inlier count and affine sanity are acceptable.

## RIFT-like Run - 2026-05-12 20:23:05

### Command

```powershell
python scripts/06_coregister_rift.py --moving 'data\processed\sar_10m.tif' --reference 'data\processed\optical_10m.tif' --output 'outputs\sar_warped_to_optical_rift_r5.tif' --metrics 'outputs\metrics_rift_r5.json' --matches 'outputs\matches_rift_r5.png' --model partial-affine --max-points 7000 --quality 0.002 --min-distance 4.0 --orientations 16 --wavelengths 4,8,16 --gamma 0.5 --patch 48 --cells 4 --ratio 0.98 --ransac-thresh 3.0 --max-displacement 35.0 --sar-log --no-mutual
```

### Metrics

- detected moving/reference: `3490` / `2943`
- matches after ratio: `1598`
- matches after spatial gate: `249`
- RANSAC inliers: `29`
- inlier ratio: `0.1165`
- RMSE: `2.1412` pixels
- near identity: `True`

### Assessment

Usable RIFT-like candidate: inlier count and affine sanity are acceptable.

## Visual Evaluation - RIFT-like R3/R5

Generated overlay and checkerboard outputs for the strongest RIFT-like candidates.

- R3 overlay after: `outputs/rift_r3/overlay_after.png`
- R3 checkerboard after: `outputs/rift_r3/checkerboard_after.png`
- R5 overlay after: `outputs/rift_r5/overlay_after.png`
- R5 checkerboard after: `outputs/rift_r5/checkerboard_after.png`
- Summary: `outputs/rift_summary.json`

Assessment: R3 is best by inlier count; R5 is best by RMSE among rounds with at least 20 inliers. Both preserve a near-identity transform, which is consistent with the geocoded input pair.


## SAR Filtering - 2026-05-12 20:31:22

- method: `gaussian`
- input: `data\processed\sar_10m.tif`
- output: `data\processed\filtered\sar_gaussian.tif`
- nonzero pct: `100.00`
- p50: `0.097851`
- p98: `0.422391`
- max: `344.234711`

## SAR Filtering - 2026-05-12 20:31:22

- method: `bilateral`
- input: `data\processed\sar_10m.tif`
- output: `data\processed\filtered\sar_bilateral.tif`
- nonzero pct: `100.00`
- p50: `0.094173`
- p98: `0.387944`
- max: `0.696801`

## SAR Filtering - 2026-05-12 20:31:22

- method: `median`
- input: `data\processed\sar_10m.tif`
- output: `data\processed\filtered\sar_median.tif`
- nonzero pct: `100.00`
- p50: `0.091932`
- p98: `0.302476`
- max: `0.696801`

## SAR Filtering - 2026-05-12 20:31:22

- method: `lee`
- input: `data\processed\sar_10m.tif`
- output: `data\processed\filtered\sar_lee.tif`
- nonzero pct: `100.00`
- p50: `0.088293`
- p98: `0.443704`
- max: `681.599182`

## SAR Filtering - 2026-05-12 20:31:23

- method: `diffusion`
- input: `data\processed\sar_10m.tif`
- output: `data\processed\filtered\sar_diffusion.tif`
- nonzero pct: `100.00`
- p50: `0.098411`
- p98: `0.429833`
- max: `2.221674`

## RIFT-like Run - 2026-05-12 20:31:58

### Command

```powershell
python scripts/06_coregister_rift.py --moving 'data\processed\sar_10m.tif' --reference 'data\processed\optical_10m.tif' --output 'outputs\filter_compare\sar_raw_rift.tif' --metrics 'outputs\filter_compare\metrics_raw_rift.json' --matches 'outputs\filter_compare\matches_raw_rift.png' --model partial-affine --max-points 7000 --quality 0.002 --min-distance 4.0 --orientations 16 --wavelengths 4,8,16 --gamma 0.5 --patch 48 --cells 4 --ratio 0.98 --ransac-thresh 4.0 --max-displacement 50.0 --sar-log --no-mutual
```

### Metrics

- detected moving/reference: `3490` / `2943`
- matches after ratio: `1598`
- matches after spatial gate: `265`
- RANSAC inliers: `42`
- inlier ratio: `0.1585`
- RMSE: `2.5178` pixels
- near identity: `True`

### Assessment

Usable RIFT-like candidate: inlier count and affine sanity are acceptable.

## RIFT-like Run - 2026-05-12 20:31:58

### Command

```powershell
python scripts/06_coregister_rift.py --moving 'data\processed\filtered\sar_gaussian.tif' --reference 'data\processed\optical_10m.tif' --output 'outputs\filter_compare\sar_gaussian_rift.tif' --metrics 'outputs\filter_compare\metrics_gaussian_rift.json' --matches 'outputs\filter_compare\matches_gaussian_rift.png' --model partial-affine --max-points 7000 --quality 0.002 --min-distance 4.0 --orientations 16 --wavelengths 4,8,16 --gamma 0.5 --patch 48 --cells 4 --ratio 0.98 --ransac-thresh 4.0 --max-displacement 50.0 --sar-log --no-mutual
```

### Metrics

- detected moving/reference: `3475` / `2943`
- matches after ratio: `1547`
- matches after spatial gate: `229`
- RANSAC inliers: `33`
- inlier ratio: `0.1441`
- RMSE: `2.5977` pixels
- near identity: `True`

### Assessment

Usable RIFT-like candidate: inlier count and affine sanity are acceptable.

## RIFT-like Run - 2026-05-12 20:31:58

### Command

```powershell
python scripts/06_coregister_rift.py --moving 'data\processed\filtered\sar_median.tif' --reference 'data\processed\optical_10m.tif' --output 'outputs\filter_compare\sar_median_rift.tif' --metrics 'outputs\filter_compare\metrics_median_rift.json' --matches 'outputs\filter_compare\matches_median_rift.png' --model partial-affine --max-points 7000 --quality 0.002 --min-distance 4.0 --orientations 16 --wavelengths 4,8,16 --gamma 0.5 --patch 48 --cells 4 --ratio 0.98 --ransac-thresh 4.0 --max-displacement 50.0 --sar-log --no-mutual
```

### Metrics

- detected moving/reference: `3352` / `2943`
- matches after ratio: `1578`
- matches after spatial gate: `266`
- RANSAC inliers: `39`
- inlier ratio: `0.1466`
- RMSE: `2.3057` pixels
- near identity: `True`

### Assessment

Usable RIFT-like candidate: inlier count and affine sanity are acceptable.

## RIFT-like Run - 2026-05-12 20:32:25

### Command

```powershell
python scripts/06_coregister_rift.py --moving 'data\processed\filtered\sar_diffusion.tif' --reference 'data\processed\optical_10m.tif' --output 'outputs\filter_compare\sar_diffusion_rift.tif' --metrics 'outputs\filter_compare\metrics_diffusion_rift.json' --matches 'outputs\filter_compare\matches_diffusion_rift.png' --model partial-affine --max-points 7000 --quality 0.002 --min-distance 4.0 --orientations 16 --wavelengths 4,8,16 --gamma 0.5 --patch 48 --cells 4 --ratio 0.98 --ransac-thresh 4.0 --max-displacement 50.0 --sar-log --no-mutual
```

### Metrics

- detected moving/reference: `3473` / `2943`
- matches after ratio: `1538`
- matches after spatial gate: `221`
- RANSAC inliers: `30`
- inlier ratio: `0.1357`
- RMSE: `2.4003` pixels
- near identity: `True`

### Assessment

Usable RIFT-like candidate: inlier count and affine sanity are acceptable.

## RIFT-like Run - 2026-05-12 20:32:25

### Command

```powershell
python scripts/06_coregister_rift.py --moving 'data\processed\filtered\sar_bilateral.tif' --reference 'data\processed\optical_10m.tif' --output 'outputs\filter_compare\sar_bilateral_rift.tif' --metrics 'outputs\filter_compare\metrics_bilateral_rift.json' --matches 'outputs\filter_compare\matches_bilateral_rift.png' --model partial-affine --max-points 7000 --quality 0.002 --min-distance 4.0 --orientations 16 --wavelengths 4,8,16 --gamma 0.5 --patch 48 --cells 4 --ratio 0.98 --ransac-thresh 4.0 --max-displacement 50.0 --sar-log --no-mutual
```

### Metrics

- detected moving/reference: `3570` / `2943`
- matches after ratio: `1571`
- matches after spatial gate: `239`
- RANSAC inliers: `40`
- inlier ratio: `0.1674`
- RMSE: `2.9071` pixels
- near identity: `True`

### Assessment

Usable RIFT-like candidate: inlier count and affine sanity are acceptable.

## RIFT-like Run - 2026-05-12 20:32:25

### Command

```powershell
python scripts/06_coregister_rift.py --moving 'data\processed\filtered\sar_lee.tif' --reference 'data\processed\optical_10m.tif' --output 'outputs\filter_compare\sar_lee_rift.tif' --metrics 'outputs\filter_compare\metrics_lee_rift.json' --matches 'outputs\filter_compare\matches_lee_rift.png' --model partial-affine --max-points 7000 --quality 0.002 --min-distance 4.0 --orientations 16 --wavelengths 4,8,16 --gamma 0.5 --patch 48 --cells 4 --ratio 0.98 --ransac-thresh 4.0 --max-displacement 50.0 --sar-log --no-mutual
```

### Metrics

- detected moving/reference: `3508` / `2943`
- matches after ratio: `1590`
- matches after spatial gate: `260`
- RANSAC inliers: `41`
- inlier ratio: `0.1577`
- RMSE: `2.7912` pixels
- near identity: `True`

### Assessment

Usable RIFT-like candidate: inlier count and affine sanity are acceptable.

## SAR Filtering + RIFT-like Comparison

Used the same RIFT-like parameters as R3 for raw SAR and five filtered SAR variants: Gaussian, median, bilateral, Lee, and anisotropic diffusion.

### Summary

- best by inliers: raw SAR, 42 inliers, RMSE 2.5178 px
- best by RMSE with at least 20 inliers: median SAR, 39 inliers, RMSE 2.3057 px
- all variants kept near-identity transforms, consistent with geocoded input data

### Files

- summary: `outputs/filter_compare/filter_rift_summary.json`
- median overlay after: `outputs/filter_compare/median_visual/overlay_after.png`
- median checkerboard after: `outputs/filter_compare/median_visual/checkerboard_after.png`

### Assessment

Filtering did not dramatically increase the number of reliable matches. Median filtering reduced RMSE while keeping enough inliers, so it is a useful comparison branch. Raw SAR remains strongest by inlier count.


## HOPC-like Run - 2026-05-12 20:38:29

### Command

```powershell
python scripts/08_coregister_hopc.py --moving 'data\processed\sar_10m.tif' --reference 'data\processed\optical_10m.tif' --output 'outputs\hopc\sar_hopc_r3.tif' --metrics 'outputs\hopc\metrics_hopc_r3.json' --matches 'outputs\hopc\matches_hopc_r3.png' --model partial-affine --max-points 7000 --quality 0.002 --min-distance 4.0 --orientations 16 --wavelengths 6,12,24 --gamma 0.5 --patch 56 --cells 4 --ratio 0.98 --ransac-thresh 4.0 --max-displacement 50.0 --sar-log --no-mutual
```

### Metrics

- moving: `data\processed\sar_10m.tif`
- detected moving/reference: `4163` / `2374`
- matches after ratio: `2016`
- matches after spatial gate: `374`
- RANSAC inliers: `40`
- inlier ratio: `0.1070`
- RMSE: `2.6272` pixels
- near identity: `True`

### Assessment

Usable HOPC-like candidate: inlier count and affine sanity are acceptable.

## HOPC-like Run - 2026-05-12 20:38:29

### Command

```powershell
python scripts/08_coregister_hopc.py --moving 'data\processed\sar_10m.tif' --reference 'data\processed\optical_10m.tif' --output 'outputs\hopc\sar_hopc_r2.tif' --metrics 'outputs\hopc\metrics_hopc_r2.json' --matches 'outputs\hopc\matches_hopc_r2.png' --model partial-affine --max-points 7000 --quality 0.002 --min-distance 4.0 --orientations 12 --wavelengths 4,8,16,24 --gamma 0.5 --patch 48 --cells 4 --ratio 0.98 --ransac-thresh 4.0 --max-displacement 50.0 --sar-log --no-mutual
```

### Metrics

- moving: `data\processed\sar_10m.tif`
- detected moving/reference: `4074` / `3670`
- matches after ratio: `2401`
- matches after spatial gate: `312`
- RANSAC inliers: `30`
- inlier ratio: `0.0962`
- RMSE: `2.7305` pixels
- near identity: `True`

### Assessment

Usable HOPC-like candidate: inlier count and affine sanity are acceptable.

## HOPC-like Run - 2026-05-12 20:38:29

### Command

```powershell
python scripts/08_coregister_hopc.py --moving 'data\processed\sar_10m.tif' --reference 'data\processed\optical_10m.tif' --output 'outputs\hopc\sar_hopc_r1.tif' --metrics 'outputs\hopc\metrics_hopc_r1.json' --matches 'outputs\hopc\matches_hopc_r1.png' --model partial-affine --max-points 7000 --quality 0.002 --min-distance 4.0 --orientations 16 --wavelengths 4,8,16 --gamma 0.5 --patch 48 --cells 4 --ratio 0.98 --ransac-thresh 4.0 --max-displacement 50.0 --sar-log --no-mutual
```

### Metrics

- moving: `data\processed\sar_10m.tif`
- detected moving/reference: `4446` / `3898`
- matches after ratio: `2153`
- matches after spatial gate: `392`
- RANSAC inliers: `71`
- inlier ratio: `0.1811`
- RMSE: `2.6298` pixels
- near identity: `True`

### Assessment

Usable HOPC-like candidate: inlier count and affine sanity are acceptable.

## HOPC-like Run - 2026-05-12 20:38:59

### Command

```powershell
python scripts/08_coregister_hopc.py --moving 'data\processed\filtered\sar_median.tif' --reference 'data\processed\optical_10m.tif' --output 'outputs\hopc\sar_median_hopc_r6.tif' --metrics 'outputs\hopc\metrics_hopc_r6_median.json' --matches 'outputs\hopc\matches_hopc_r6_median.png' --model partial-affine --max-points 7000 --quality 0.002 --min-distance 4.0 --orientations 16 --wavelengths 4,8,16 --gamma 0.5 --patch 48 --cells 4 --ratio 0.98 --ransac-thresh 4.0 --max-displacement 50.0 --sar-log --no-mutual
```

### Metrics

- moving: `data\processed\filtered\sar_median.tif`
- detected moving/reference: `4382` / `3898`
- matches after ratio: `2194`
- matches after spatial gate: `373`
- RANSAC inliers: `54`
- inlier ratio: `0.1448`
- RMSE: `2.7477` pixels
- near identity: `True`

### Assessment

Usable HOPC-like candidate: inlier count and affine sanity are acceptable.

## HOPC-like Run - 2026-05-12 20:38:59

### Command

```powershell
python scripts/08_coregister_hopc.py --moving 'data\processed\sar_10m.tif' --reference 'data\processed\optical_10m.tif' --output 'outputs\hopc\sar_hopc_r4.tif' --metrics 'outputs\hopc\metrics_hopc_r4.json' --matches 'outputs\hopc\matches_hopc_r4.png' --model partial-affine --max-points 7000 --quality 0.002 --min-distance 4.0 --orientations 16 --wavelengths 4,8,16 --gamma 0.5 --patch 48 --cells 4 --ratio 0.98 --ransac-thresh 3.0 --max-displacement 35.0 --sar-log --no-mutual
```

### Metrics

- moving: `data\processed\sar_10m.tif`
- detected moving/reference: `4446` / `3898`
- matches after ratio: `2153`
- matches after spatial gate: `351`
- RANSAC inliers: `43`
- inlier ratio: `0.1225`
- RMSE: `1.9888` pixels
- near identity: `True`

### Assessment

Usable HOPC-like candidate: inlier count and affine sanity are acceptable.

## HOPC-like Run - 2026-05-12 20:38:59

### Command

```powershell
python scripts/08_coregister_hopc.py --moving 'data\processed\sar_10m.tif' --reference 'data\processed\optical_10m.tif' --output 'outputs\hopc\sar_hopc_r5.tif' --metrics 'outputs\hopc\metrics_hopc_r5.json' --matches 'outputs\hopc\matches_hopc_r5.png' --model partial-affine --max-points 7000 --quality 0.002 --min-distance 4.0 --orientations 16 --wavelengths 4,8,16 --gamma 0.5 --patch 48 --cells 4 --ratio 0.99 --ransac-thresh 4.0 --max-displacement 50.0 --sar-log
```

### Metrics

- moving: `data\processed\sar_10m.tif`
- detected moving/reference: `4446` / `3898`
- matches after ratio: `389`
- matches after spatial gate: `93`
- RANSAC inliers: `20`
- inlier ratio: `0.2151`
- RMSE: `2.5203` pixels
- near identity: `True`

### Assessment

Usable HOPC-like candidate: inlier count and affine sanity are acceptable.

## HOPC-like Summary

Implemented a HOPC-like phase-congruency orientation histogram descriptor and ran six rounds, including one median-filtered SAR branch.

### Best Results

- best by inliers: HOPC R1, 71 inliers, RMSE 2.6298 px
- best by RMSE with at least 20 inliers: HOPC R4, 43 inliers, RMSE 1.9888 px
- median SAR branch: 54 inliers, RMSE 2.7477 px

### Files

- summary: `outputs/hopc/hopc_summary.json`
- R1 overlay after: `outputs/hopc/visual_r1/overlay_after.png`
- R4 overlay after: `outputs/hopc/visual_r4/overlay_after.png`

### Assessment

HOPC-like is currently the strongest traditional descriptor in this project. It improves inlier count over RIFT-like and gives the lowest RMSE so far in R4.


## PCSD-like Run - 2026-05-12 20:47:20

### Command

```powershell
python scripts/09_coregister_pcsd.py --moving 'data\processed\sar_10m.tif' --reference 'data\processed\optical_10m.tif' --output 'outputs\pcsd\sar_pcsd_r3.tif' --metrics 'outputs\pcsd\metrics_pcsd_r3.json' --matches 'outputs\pcsd\matches_pcsd_r3.png' --model partial-affine --max-points 7000 --quality 0.002 --min-distance 4.0 --orientations 16 --wavelengths 6,12,24 --gamma 0.5 --patch 56 --cells 4 --ratio 0.98 --ransac-thresh 4.0 --max-displacement 50.0 --sar-log --no-mutual
```

### Metrics

- moving: `data\processed\sar_10m.tif`
- detected moving/reference: `5643` / `5145`
- matches after ratio: `600`
- matches after spatial gate: `16`
- RANSAC inliers: `4`
- inlier ratio: `0.2500`
- RMSE: `1.7674` pixels
- near identity: `False`

### Assessment

Not reliable as a final registration transform; keep as an experimental round.

## PCSD-like Run - 2026-05-12 20:47:20

### Command

```powershell
python scripts/09_coregister_pcsd.py --moving 'data\processed\sar_10m.tif' --reference 'data\processed\optical_10m.tif' --output 'outputs\pcsd\sar_pcsd_r1.tif' --metrics 'outputs\pcsd\metrics_pcsd_r1.json' --matches 'outputs\pcsd\matches_pcsd_r1.png' --model partial-affine --max-points 7000 --quality 0.002 --min-distance 4.0 --orientations 16 --wavelengths 4,8,16 --gamma 0.5 --patch 48 --cells 4 --ratio 0.98 --ransac-thresh 4.0 --max-displacement 50.0 --sar-log --no-mutual
```

### Metrics

- moving: `data\processed\sar_10m.tif`
- detected moving/reference: `5759` / `5520`
- matches after ratio: `612`
- matches after spatial gate: `23`
- RANSAC inliers: `7`
- inlier ratio: `0.3043`
- RMSE: `2.0437` pixels
- near identity: `False`

### Assessment

Not reliable as a final registration transform; keep as an experimental round.

## PCSD-like Run - 2026-05-12 20:47:20

### Command

```powershell
python scripts/09_coregister_pcsd.py --moving 'data\processed\sar_10m.tif' --reference 'data\processed\optical_10m.tif' --output 'outputs\pcsd\sar_pcsd_r2.tif' --metrics 'outputs\pcsd\metrics_pcsd_r2.json' --matches 'outputs\pcsd\matches_pcsd_r2.png' --model partial-affine --max-points 7000 --quality 0.002 --min-distance 4.0 --orientations 12 --wavelengths 4,8,16,24 --gamma 0.5 --patch 48 --cells 4 --ratio 0.98 --ransac-thresh 4.0 --max-displacement 50.0 --sar-log --no-mutual
```

### Metrics

- moving: `data\processed\sar_10m.tif`
- detected moving/reference: `5733` / `5552`
- matches after ratio: `936`
- matches after spatial gate: `44`
- RANSAC inliers: `8`
- inlier ratio: `0.1818`
- RMSE: `2.0587` pixels
- near identity: `False`

### Assessment

Not reliable as a final registration transform; keep as an experimental round.

## PCSD-like Run - 2026-05-12 20:48:14

### Command

```powershell
python scripts/09_coregister_pcsd.py --moving 'data\processed\filtered\sar_diffusion.tif' --reference 'data\processed\optical_10m.tif' --output 'outputs\pcsd\sar_diffusion_pcsd_r6.tif' --metrics 'outputs\pcsd\metrics_pcsd_r6_diffusion.json' --matches 'outputs\pcsd\matches_pcsd_r6_diffusion.png' --model partial-affine --max-points 7000 --quality 0.002 --min-distance 4.0 --orientations 16 --wavelengths 4,8,16 --gamma 0.5 --patch 48 --cells 4 --ratio 0.99 --ransac-thresh 3.0 --max-displacement 30.0 --sar-log
```

### Metrics

- moving: `data\processed\filtered\sar_diffusion.tif`
- detected moving/reference: `3791` / `5520`
- matches after ratio: `83`
- matches after spatial gate: `6`
- RANSAC inliers: `3`
- inlier ratio: `0.5000`
- RMSE: `0.8117` pixels
- near identity: `True`

### Assessment

Not reliable as a final registration transform; keep as an experimental round.

## PCSD-like Run - 2026-05-12 20:48:18

### Command

```powershell
python scripts/09_coregister_pcsd.py --moving 'data\processed\filtered\sar_median.tif' --reference 'data\processed\optical_10m.tif' --output 'outputs\pcsd\sar_median_pcsd_r5.tif' --metrics 'outputs\pcsd\metrics_pcsd_r5_median.json' --matches 'outputs\pcsd\matches_pcsd_r5_median.png' --model partial-affine --max-points 7000 --quality 0.002 --min-distance 4.0 --orientations 16 --wavelengths 4,8,16 --gamma 0.5 --patch 48 --cells 4 --ratio 0.99 --ransac-thresh 3.0 --max-displacement 30.0 --sar-log
```

### Metrics

- moving: `data\processed\filtered\sar_median.tif`
- detected moving/reference: `5314` / `5520`
- matches after ratio: `109`
- matches after spatial gate: `10`
- RANSAC inliers: `4`
- inlier ratio: `0.4000`
- RMSE: `0.8929` pixels
- near identity: `False`

### Assessment

Not reliable as a final registration transform; keep as an experimental round.

## PCSD-like Run - 2026-05-12 20:49:16

### Command

```powershell
python scripts/09_coregister_pcsd.py --moving 'data\processed\filtered\sar_diffusion.tif' --reference 'data\processed\optical_10m.tif' --output 'outputs\pcsd\sar_diffusion_pcsd_r8.tif' --metrics 'outputs\pcsd\metrics_pcsd_r8_diffusion.json' --matches 'outputs\pcsd\matches_pcsd_r8_diffusion.png' --model partial-affine --max-points 7000 --quality 0.001 --min-distance 4.0 --orientations 16 --wavelengths 4,8,16 --gamma 0.5 --patch 40 --cells 4 --ratio 0.99 --ransac-thresh 3.0 --max-displacement 25.0 --sar-log --no-mutual
```

### Metrics

- moving: `data\processed\filtered\sar_diffusion.tif`
- detected moving/reference: `4466` / `6048`
- matches after ratio: `1588`
- matches after spatial gate: `43`
- RANSAC inliers: `7`
- inlier ratio: `0.1628`
- RMSE: `1.6671` pixels
- near identity: `True`

### Assessment

Not reliable as a final registration transform; keep as an experimental round.

## PCSD-like Run - 2026-05-12 20:49:20

### Command

```powershell
python scripts/09_coregister_pcsd.py --moving 'data\processed\filtered\sar_median.tif' --reference 'data\processed\optical_10m.tif' --output 'outputs\pcsd\sar_median_pcsd_r9.tif' --metrics 'outputs\pcsd\metrics_pcsd_r9_median.json' --matches 'outputs\pcsd\matches_pcsd_r9_median.png' --model partial-affine --max-points 7000 --quality 0.001 --min-distance 4.0 --orientations 16 --wavelengths 4,8,16 --gamma 0.5 --patch 40 --cells 4 --ratio 0.99 --ransac-thresh 3.0 --max-displacement 25.0 --sar-log --no-mutual
```

### Metrics

- moving: `data\processed\filtered\sar_median.tif`
- detected moving/reference: `5856` / `6048`
- matches after ratio: `2148`
- matches after spatial gate: `46`
- RANSAC inliers: `11`
- inlier ratio: `0.2391`
- RMSE: `1.7240` pixels
- near identity: `True`

### Assessment

Usable PCSD-like candidate: inlier count and affine sanity are acceptable.

## PCSD-like Run - 2026-05-12 20:49:21

### Command

```powershell
python scripts/09_coregister_pcsd.py --moving 'data\processed\sar_10m.tif' --reference 'data\processed\optical_10m.tif' --output 'outputs\pcsd\sar_pcsd_r7.tif' --metrics 'outputs\pcsd\metrics_pcsd_r7.json' --matches 'outputs\pcsd\matches_pcsd_r7.png' --model partial-affine --max-points 7000 --quality 0.001 --min-distance 4.0 --orientations 16 --wavelengths 4,8,16 --gamma 0.5 --patch 40 --cells 4 --ratio 0.99 --ransac-thresh 3.0 --max-displacement 25.0 --sar-log --no-mutual
```

### Metrics

- moving: `data\processed\sar_10m.tif`
- detected moving/reference: `6132` / `6048`
- matches after ratio: `2178`
- matches after spatial gate: `16`
- RANSAC inliers: `4`
- inlier ratio: `0.2500`
- RMSE: `1.1815` pixels
- near identity: `True`

### Assessment

Not reliable as a final registration transform; keep as an experimental round.

## PCSD-like Summary

Implemented a PCSD-like phase-congruency structural descriptor combining local phase-congruency orientation histograms, gradient orientation histograms, and cell-level structure statistics.

### Best Result

- best usable: median SAR + PCSD R9, 11 inliers, RMSE 1.7240 px, near-identity transform
- several low-RMSE rounds were rejected because they had too few inliers or non-near-identity transforms

### Files

- script: `scripts/09_coregister_pcsd.py`
- summary: `outputs/pcsd/pcsd_summary.json`
- best metrics: `outputs/pcsd/metrics_pcsd_r9_median.json`
- best overlay after: `outputs/pcsd/visual_r9_median/overlay_after.png`

### Assessment

This PCSD-like implementation did not outperform HOPC-like. It can be reported as an attempted PCSD-inspired branch: low RMSE is possible, but stable inlier count is weak on this AOI. HOPC-like remains the strongest traditional method so far.


## HOPC-like Run - 2026-05-12 20:54:43

### Command

```powershell
python scripts/08_coregister_hopc.py --moving 'data\processed\sar_10m.tif' --reference 'data\processed\optical_10m.tif' --output 'outputs\hopc_opt\sar_hopc_opt_r2.tif' --metrics 'outputs\hopc_opt\metrics_hopc_opt_r2.json' --matches 'outputs\hopc_opt\matches_hopc_opt_r2.png' --model partial-affine --max-points 7000 --quality 0.002 --min-distance 4.0 --orientations 16 --wavelengths 4,8,16 --gamma 0.5 --patch 48 --cells 4 --ratio 0.98 --ransac-thresh 2.0 --max-displacement 35.0 --sar-log --no-mutual
```

### Metrics

- moving: `data\processed\sar_10m.tif`
- detected moving/reference: `4446` / `3898`
- matches after ratio: `2153`
- matches after spatial gate: `351`
- RANSAC inliers: `27`
- inlier ratio: `0.0769`
- RMSE: `1.2625` pixels
- near identity: `True`

### Assessment

Usable HOPC-like candidate: inlier count and affine sanity are acceptable.

## HOPC-like Run - 2026-05-12 20:54:43

### Command

```powershell
python scripts/08_coregister_hopc.py --moving 'data\processed\sar_10m.tif' --reference 'data\processed\optical_10m.tif' --output 'outputs\hopc_opt\sar_hopc_opt_r1.tif' --metrics 'outputs\hopc_opt\metrics_hopc_opt_r1.json' --matches 'outputs\hopc_opt\matches_hopc_opt_r1.png' --model partial-affine --max-points 7000 --quality 0.002 --min-distance 4.0 --orientations 16 --wavelengths 4,8,16 --gamma 0.5 --patch 48 --cells 4 --ratio 0.98 --ransac-thresh 2.5 --max-displacement 35.0 --sar-log --no-mutual
```

### Metrics

- moving: `data\processed\sar_10m.tif`
- detected moving/reference: `4446` / `3898`
- matches after ratio: `2153`
- matches after spatial gate: `351`
- RANSAC inliers: `33`
- inlier ratio: `0.0940`
- RMSE: `1.7606` pixels
- near identity: `True`

### Assessment

Usable HOPC-like candidate: inlier count and affine sanity are acceptable.

## HOPC-like Run - 2026-05-12 20:54:43

### Command

```powershell
python scripts/08_coregister_hopc.py --moving 'data\processed\sar_10m.tif' --reference 'data\processed\optical_10m.tif' --output 'outputs\hopc_opt\sar_hopc_opt_r3.tif' --metrics 'outputs\hopc_opt\metrics_hopc_opt_r3.json' --matches 'outputs\hopc_opt\matches_hopc_opt_r3.png' --model partial-affine --max-points 7000 --quality 0.002 --min-distance 4.0 --orientations 16 --wavelengths 4,8,16 --gamma 0.5 --patch 48 --cells 4 --ratio 0.98 --ransac-thresh 3.0 --max-displacement 25.0 --sar-log --no-mutual
```

### Metrics

- moving: `data\processed\sar_10m.tif`
- detected moving/reference: `4446` / `3898`
- matches after ratio: `2153`
- matches after spatial gate: `324`
- RANSAC inliers: `43`
- inlier ratio: `0.1327`
- RMSE: `1.8741` pixels
- near identity: `True`

### Assessment

Usable HOPC-like candidate: inlier count and affine sanity are acceptable.

## HOPC-like Run - 2026-05-12 20:55:14

### Command

```powershell
python scripts/08_coregister_hopc.py --moving 'data\processed\sar_10m.tif' --reference 'data\processed\optical_10m.tif' --output 'outputs\hopc_opt\sar_hopc_opt_r5.tif' --metrics 'outputs\hopc_opt\metrics_hopc_opt_r5.json' --matches 'outputs\hopc_opt\matches_hopc_opt_r5.png' --model partial-affine --max-points 7000 --quality 0.002 --min-distance 4.0 --orientations 16 --wavelengths 4,8,16 --gamma 0.5 --patch 40 --cells 4 --ratio 0.98 --ransac-thresh 2.5 --max-displacement 35.0 --sar-log --no-mutual
```

### Metrics

- moving: `data\processed\sar_10m.tif`
- detected moving/reference: `4572` / `4017`
- matches after ratio: `2553`
- matches after spatial gate: `328`
- RANSAC inliers: `28`
- inlier ratio: `0.0854`
- RMSE: `1.8450` pixels
- near identity: `True`

### Assessment

Usable HOPC-like candidate: inlier count and affine sanity are acceptable.

## HOPC-like Run - 2026-05-12 20:55:14

### Command

```powershell
python scripts/08_coregister_hopc.py --moving 'data\processed\filtered\sar_median.tif' --reference 'data\processed\optical_10m.tif' --output 'outputs\hopc_opt\sar_median_hopc_opt_r6.tif' --metrics 'outputs\hopc_opt\metrics_hopc_opt_r6_median.json' --matches 'outputs\hopc_opt\matches_hopc_opt_r6_median.png' --model partial-affine --max-points 7000 --quality 0.002 --min-distance 4.0 --orientations 16 --wavelengths 4,8,16 --gamma 0.5 --patch 48 --cells 4 --ratio 0.98 --ransac-thresh 2.5 --max-displacement 35.0 --sar-log --no-mutual
```

### Metrics

- moving: `data\processed\filtered\sar_median.tif`
- detected moving/reference: `4382` / `3898`
- matches after ratio: `2194`
- matches after spatial gate: `343`
- RANSAC inliers: `30`
- inlier ratio: `0.0875`
- RMSE: `1.5038` pixels
- near identity: `True`

### Assessment

Usable HOPC-like candidate: inlier count and affine sanity are acceptable.

## HOPC-like Run - 2026-05-12 20:55:15

### Command

```powershell
python scripts/08_coregister_hopc.py --moving 'data\processed\sar_10m.tif' --reference 'data\processed\optical_10m.tif' --output 'outputs\hopc_opt\sar_hopc_opt_r4.tif' --metrics 'outputs\hopc_opt\metrics_hopc_opt_r4.json' --matches 'outputs\hopc_opt\matches_hopc_opt_r4.png' --model partial-affine --max-points 7000 --quality 0.002 --min-distance 4.0 --orientations 20 --wavelengths 4,8,16 --gamma 0.5 --patch 48 --cells 4 --ratio 0.98 --ransac-thresh 2.5 --max-displacement 35.0 --sar-log --no-mutual
```

### Metrics

- moving: `data\processed\sar_10m.tif`
- detected moving/reference: `4625` / `4213`
- matches after ratio: `2211`
- matches after spatial gate: `287`
- RANSAC inliers: `27`
- inlier ratio: `0.0941`
- RMSE: `1.5401` pixels
- near identity: `True`

### Assessment

Usable HOPC-like candidate: inlier count and affine sanity are acceptable.

## HOPC-like Run - 2026-05-12 20:55:48

### Command

```powershell
python scripts/08_coregister_hopc.py --moving 'data\processed\sar_10m.tif' --reference 'data\processed\optical_10m.tif' --output 'outputs\hopc_opt\sar_hopc_opt_r8.tif' --metrics 'outputs\hopc_opt\metrics_hopc_opt_r8.json' --matches 'outputs\hopc_opt\matches_hopc_opt_r8.png' --model partial-affine --max-points 7000 --quality 0.002 --min-distance 4.0 --orientations 16 --wavelengths 4,8,16 --gamma 0.5 --patch 48 --cells 4 --ratio 0.98 --ransac-thresh 2.8 --max-displacement 30.0 --sar-log --no-mutual
```

### Metrics

- moving: `data\processed\sar_10m.tif`
- detected moving/reference: `4446` / `3898`
- matches after ratio: `2153`
- matches after spatial gate: `342`
- RANSAC inliers: `37`
- inlier ratio: `0.1082`
- RMSE: `1.7846` pixels
- near identity: `True`

### Assessment

Usable HOPC-like candidate: inlier count and affine sanity are acceptable.

## HOPC-like Run - 2026-05-12 20:55:48

### Command

```powershell
python scripts/08_coregister_hopc.py --moving 'data\processed\sar_10m.tif' --reference 'data\processed\optical_10m.tif' --output 'outputs\hopc_opt\sar_hopc_opt_r7.tif' --metrics 'outputs\hopc_opt\metrics_hopc_opt_r7.json' --matches 'outputs\hopc_opt\matches_hopc_opt_r7.png' --model partial-affine --max-points 7000 --quality 0.002 --min-distance 4.0 --orientations 16 --wavelengths 4,8,16 --gamma 0.5 --patch 48 --cells 4 --ratio 0.98 --ransac-thresh 2.2 --max-displacement 30.0 --sar-log --no-mutual
```

### Metrics

- moving: `data\processed\sar_10m.tif`
- detected moving/reference: `4446` / `3898`
- matches after ratio: `2153`
- matches after spatial gate: `342`
- RANSAC inliers: `25`
- inlier ratio: `0.0731`
- RMSE: `1.4065` pixels
- near identity: `True`

### Assessment

Usable HOPC-like candidate: inlier count and affine sanity are acceptable.

## HOPC-like Run - 2026-05-12 20:55:48

### Command

```powershell
python scripts/08_coregister_hopc.py --moving 'data\processed\filtered\sar_median.tif' --reference 'data\processed\optical_10m.tif' --output 'outputs\hopc_opt\sar_median_hopc_opt_r9.tif' --metrics 'outputs\hopc_opt\metrics_hopc_opt_r9_median.json' --matches 'outputs\hopc_opt\matches_hopc_opt_r9_median.png' --model partial-affine --max-points 7000 --quality 0.002 --min-distance 4.0 --orientations 16 --wavelengths 4,8,16 --gamma 0.5 --patch 48 --cells 4 --ratio 0.98 --ransac-thresh 2.2 --max-displacement 30.0 --sar-log --no-mutual
```

### Metrics

- moving: `data\processed\filtered\sar_median.tif`
- detected moving/reference: `4382` / `3898`
- matches after ratio: `2194`
- matches after spatial gate: `329`
- RANSAC inliers: `25`
- inlier ratio: `0.0760`
- RMSE: `1.4136` pixels
- near identity: `True`

### Assessment

Usable HOPC-like candidate: inlier count and affine sanity are acceptable.

## HOPC-like Focused Optimization Summary

Ran a focused optimization around the previous HOPC R4 parameters: tighter RANSAC thresholds, smaller spatial gates, patch/orientation variants, and median-filtered SAR variants.

### Best Optimized Results

- best RMSE with at least 20 inliers: HOPC opt R2, 27 inliers, RMSE 1.2625 px
- best balanced result with at least 35 inliers: HOPC opt R8, 37 inliers, RMSE 1.7846 px
- best inlier count in this optimization: HOPC opt R3, 43 inliers, RMSE 1.8741 px

### Files

- summary: `outputs/hopc_opt/hopc_opt_summary.json`
- R2 overlay after: `outputs/hopc_opt/visual_r2/overlay_after.png`
- R8 overlay after: `outputs/hopc_opt/visual_r8/overlay_after.png`

### Assessment

The optimized HOPC-like runs improve the earlier HOPC R4 RMSE from about 1.99 px to 1.26 px when accepting 27 inliers. For a more balanced final result, R8 keeps 37 inliers with 1.78 px RMSE. Both transforms remain near identity.

