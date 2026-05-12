# SAR-Optical 跨模态配准方法与结果总览

本文档汇总当前项目中已经使用过的全部 SAR-Optical 配准方法、对应脚本、主要输出文件和实验结果。

实验区域为成都天府国际机场核心区域。SAR 数据为 Sentinel-1A GRD，Optical 数据为 Sentinel-2C L2A。两者已经统一到同一 CRS、同一分辨率、同一空间网格。

## 1. 数据与预处理

### 使用数据

SAR:

```text
Sentinel-1A GRD
S1A_IW_GRDH_1SDV_20250805T110854_20250805T110919_060400_0781F8_677D.zip
Date: 2025-08-05
Polarization: VV + VH
Used band: VV
```

Optical:

```text
Sentinel-2C L2A
S2C_MSIL2A_20250803T033601_N0511_R061_T48RVU_20250803T085013.SAFE.zip
Date: 2025-08-03
Used bands: B4 / B3 / B2
```

时间间隔约 2.31 天。

### AOI

```text
Chengdu Tianfu International Airport core AOI
lon/lat: 104.40, 30.27, 104.48, 30.32
CRS: EPSG:32648
Resolution: 10 m
Size: 772 x 558
```

### 数据质量

```text
SAR valid pixels: 100%
Optical valid pixels: 100%
Same grid: true
```

数据质量文件:

```text
outputs/data_quality.json
outputs/by_method/00_data_and_preprocessing/
```

### 预处理脚本

```text
scripts/01_preprocess_sar.py
scripts/02_preprocess_optical.py
scripts/04_evaluate_overlay.py
```

SAR 预处理流程:

```text
Sentinel-1 GRD
→ Apply Orbit File
→ Remove GRD Border Noise
→ Radiometric Calibration
→ Terrain Correction
→ VV sigma0 GeoTIFF
→ AOI crop
```

Optical 预处理流程:

```text
Sentinel-2 L2A
→ Extract B4/B3/B2
→ RGB GeoTIFF
→ Reproject/resample to SAR grid
```

## 2. 已使用方法总表

| 类别 | 方法 | 脚本 | 主要结果 |
|---|---|---|---|
| Baseline | SIFT / ORB / AKAZE | `scripts/03_coregister_opencv.py` | 普通局部特征效果较弱 |
| Structural descriptor | CFOG-like | `scripts/05_coregister_cfog.py` | 有改善，但内点较少 |
| Structural descriptor | RIFT-like | `scripts/06_coregister_rift.py` | 明显优于 SIFT/CFOG-like |
| SAR filtering | Gaussian / Median / Bilateral / Lee / Diffusion + RIFT-like | `scripts/07_filter_sar.py` + `scripts/06_coregister_rift.py` | Median filtering 可降低 RMSE，但提升有限 |
| Structural descriptor | HOPC-like | `scripts/08_coregister_hopc.py` | 当前最强传统方法 |
| Structural descriptor | PCSD-like | `scripts/09_coregister_pcsd.py` | 可用结果较弱，没有超过 HOPC-like |
| Optimized | HOPC-like focused optimization | `scripts/08_coregister_hopc.py` | 当前最终推荐结果 |

## 3. OpenCV Baseline

### 方法

使用传统 OpenCV 特征：

```text
SIFT
ORB
AKAZE
```

流程:

```text
SAR VV intensity
→ log(1 + SAR)
→ uint8 normalization
→ feature detection and description
→ descriptor matching
→ Lowe ratio test
→ RANSAC
→ affine / homography estimation
→ warp SAR to Optical
```

### 结果

SIFT affine:

```text
matches_after_ratio: 28
RANSAC inliers: 5
inlier_ratio: 0.1786
```

结论:

```text
普通 SIFT / ORB / AKAZE 对 SAR-Optical 跨模态图像不够稳定。
```

归档目录:

```text
outputs/by_method/01_opencv_baseline_sift_orb_akaze/
```

## 4. CFOG-like

### 方法

CFOG-like 使用梯度方向结构描述子：

```text
SAR / Optical
→ CLAHE + Gaussian smoothing
→ gradient magnitude / orientation
→ multi-orientation gradient channels
→ local grid descriptor
→ descriptor matching
→ spatial gate
→ RANSAC partial-affine
```

### 最佳结果

```text
CFOG-like R5
inliers: 20
RMSE: 2.49 px
near-identity transform: true
```

结论:

```text
CFOG-like 明显优于普通 SIFT，但内点数量仍有限。
```

主要文件:

```text
outputs/cfog_summary.json
outputs/metrics_cfog_r5.json
outputs/cfog_r5/overlay_after.png
outputs/by_method/02_cfog_like/
```

## 5. RIFT-like

### 方法

RIFT-like 使用 MIM 方向索引图思想：

```text
SAR / Optical
→ CLAHE
→ multi-scale multi-orientation Gabor response
→ MIM: Maximum Index Map
→ local MIM histogram descriptor
→ descriptor matching
→ spatial gate
→ RANSAC partial-affine
```

### 最佳结果

RIFT R3:

```text
inliers: 42
RMSE: 2.52 px
near-identity transform: true
```

RIFT R5:

```text
inliers: 29
RMSE: 2.14 px
near-identity transform: true
```

结论:

```text
RIFT-like 明显优于 CFOG-like 和 SIFT baseline。
```

主要文件:

```text
outputs/rift_summary.json
outputs/metrics_rift_r3.json
outputs/metrics_rift_r5.json
outputs/rift_r3/overlay_after.png
outputs/rift_r5/overlay_after.png
outputs/by_method/03_rift_like/
```

## 6. SAR Filtering + RIFT-like

### 方法

测试了以下 SAR 滤波方法：

```text
Gaussian
Median
Bilateral
Lee
Anisotropic diffusion
```

然后使用相同 RIFT-like 参数进行配准。

### 结果

| SAR 输入 | Inliers | RMSE px | 说明 |
|---|---:|---:|---|
| raw | 42 | 2.52 | 内点最多 |
| gaussian | 33 | 2.60 | 一般 |
| median | 39 | 2.31 | RMSE 改善 |
| bilateral | 40 | 2.91 | 内点多但误差高 |
| Lee | 41 | 2.79 | 内点多但误差高 |
| diffusion | 30 | 2.40 | 误差有改善 |

结论:

```text
滤波没有压倒性提升。
Median filtering 对 RMSE 有一定改善，但 raw SAR 仍保持最高内点数。
```

主要文件:

```text
outputs/filter_compare/filter_rift_summary.json
outputs/filter_compare/metrics_median_rift.json
outputs/filter_compare/median_visual/overlay_after.png
outputs/by_method/04_sar_filtering_plus_rift/
```

## 7. HOPC-like

### 方法

HOPC-like 使用相位一致性方向直方图：

```text
SAR / Optical
→ CLAHE
→ multi-scale multi-orientation Gabor even/odd response
→ approximate phase congruency
→ principal orientation map
→ local orientation histogram descriptor
→ descriptor matching
→ spatial gate
→ RANSAC partial-affine
```

### 初始最佳结果

HOPC R1:

```text
inliers: 71
RMSE: 2.63 px
near-identity transform: true
```

HOPC R4:

```text
inliers: 43
RMSE: 1.99 px
near-identity transform: true
```

结论:

```text
HOPC-like 是当前最强的传统结构描述子方法。
```

主要文件:

```text
outputs/hopc/hopc_summary.json
outputs/hopc/metrics_hopc_r1.json
outputs/hopc/metrics_hopc_r4.json
outputs/hopc/visual_r1/overlay_after.png
outputs/hopc/visual_r4/overlay_after.png
outputs/by_method/05_hopc_like/
```

## 8. PCSD-like

### 方法

PCSD-like 组合了相位一致性、方向直方图和结构统计量：

```text
SAR / Optical
→ CLAHE
→ multi-scale multi-orientation Gabor even/odd response
→ phase congruency map
→ phase congruency orientation
→ gradient magnitude / orientation
→ PC orientation histogram
→ gradient orientation histogram
→ cell structure statistics
→ descriptor matching
→ spatial gate
→ RANSAC partial-affine
```

### 最佳可用结果

Median SAR + PCSD R9:

```text
inliers: 11
RMSE: 1.72 px
near-identity transform: true
```

结论:

```text
PCSD-like 可以取得较低 RMSE，但稳定内点数量不足。
当前没有超过 HOPC-like。
```

主要文件:

```text
outputs/pcsd/pcsd_summary.json
outputs/pcsd/metrics_pcsd_r9_median.json
outputs/pcsd/visual_r9_median/overlay_after.png
outputs/by_method/06_pcsd_like/
```

## 9. HOPC-like Focused Optimization

### 优化目标

在 HOPC-like 基础上继续调参：

```text
RANSAC threshold
max displacement
patch size
orientation bins
raw SAR / median SAR
```

### 最佳结果

精度最优:

```text
HOPC opt R2
inliers: 27
RMSE: 1.26 px
near-identity transform: true
```

均衡结果:

```text
HOPC opt R8
inliers: 37
RMSE: 1.78 px
near-identity transform: true
```

内点最多:

```text
HOPC opt R3
inliers: 43
RMSE: 1.87 px
near-identity transform: true
```

结论:

```text
HOPC opt R2 是当前 RMSE 最低的结果。
HOPC opt R8 是当前更适合作为主结果的均衡版本。
```

主要文件:

```text
outputs/hopc_opt/hopc_opt_summary.json
outputs/hopc_opt/metrics_hopc_opt_r2.json
outputs/hopc_opt/metrics_hopc_opt_r8.json
outputs/hopc_opt/visual_r2/overlay_after.png
outputs/hopc_opt/visual_r8/overlay_after.png
outputs/by_method/07_hopc_optimized/
```

## 10. 总体结果排序

按整体表现排序：

```text
SIFT / ORB / AKAZE baseline
< CFOG-like
< RIFT-like
< PCSD-like
< HOPC-like
< Optimized HOPC-like
```

更严格地说：

```text
PCSD-like 的 RMSE 可以较低，但内点太少，因此不作为主结果。
Optimized HOPC-like 是当前最终推荐方法。
```

## 11. 总结果表

| 方法 | 最佳版本 | Inliers | RMSE px | 是否推荐 |
|---|---|---:|---:|---|
| SIFT baseline | SIFT affine | 5 | 1.18 | 否，内点太少 |
| CFOG-like | R5 | 20 | 2.49 | 可作为改进对比 |
| RIFT-like | R3 / R5 | 42 / 29 | 2.52 / 2.14 | 可作为强对比 |
| Median + RIFT-like | median branch | 39 | 2.31 | 可作为滤波对比 |
| HOPC-like | R1 / R4 | 71 / 43 | 2.63 / 1.99 | 推荐 |
| PCSD-like | median R9 | 11 | 1.72 | 不作为主结果 |
| Optimized HOPC-like | R2 | 27 | 1.26 | 精度最优 |
| Optimized HOPC-like | R8 | 37 | 1.78 | 主结果推荐 |

## 12. 最终推荐

最终建议使用两个结果：

### 主结果

```text
HOPC opt R8
inliers: 37
RMSE: 1.78 px
```

理由：

```text
内点数量相对稳定，RMSE 明显优于普通 HOPC/RIFT/CFOG，变换矩阵接近单位阵。
```

文件：

```text
outputs/hopc_opt/metrics_hopc_opt_r8.json
outputs/hopc_opt/sar_hopc_opt_r8.tif
outputs/hopc_opt/visual_r8/overlay_after.png
outputs/hopc_opt/visual_r8/checkerboard_after.png
```

### 精度最优结果

```text
HOPC opt R2
inliers: 27
RMSE: 1.26 px
```

理由：

```text
当前所有可用结果中 RMSE 最低，但内点数比 R8 少。
```

文件：

```text
outputs/hopc_opt/metrics_hopc_opt_r2.json
outputs/hopc_opt/sar_hopc_opt_r2.tif
outputs/hopc_opt/visual_r2/overlay_after.png
outputs/hopc_opt/visual_r2/checkerboard_after.png
```

## 13. 归档目录

所有方法已经按方法归档到：

```text
outputs/by_method/
```

归档索引：

```text
outputs/by_method/README.md
```

实验过程日志：

```text
EXPERIMENT_LOG.md
outputs/by_method/99_logs_and_reports/EXPERIMENT_LOG.md
```

## 14. 注意事项

当前所有结构方法都应表述为：

```text
CFOG-like
RIFT-like
HOPC-like
PCSD-like
```

原因是这些方法是工程近似实现，用于验证结构描述思想，并不是论文官方代码或严格公式级复现。

推荐论文/汇报表述：

```text
本文参考 CFOG、RIFT、HOPC 和 PCSD 的结构描述思想，实现了若干跨模态结构特征描述方法，并在 Sentinel-1 SAR 与 Sentinel-2 Optical 图像配准任务上进行了对比实验。
实验结果表明，HOPC-like 方法在当前数据上取得最优综合表现。
```

