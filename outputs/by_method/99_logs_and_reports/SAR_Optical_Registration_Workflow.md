# SAR-Optical 跨模态图像配准处理流程

本文档说明当前项目中 Sentinel-1 SAR 与 Sentinel-2 Optical 图像配准的完整处理流程。

研究目标是把同一地理区域内的 Sentinel-1 SAR 图像和 Sentinel-2 光学图像统一到同一空间网格，并尝试使用传统 OpenCV 特征方法完成 baseline 配准，为后续城市形变监测、建筑变化检测或更复杂的跨模态配准方法做准备。

## 1. 数据输入

当前使用的数据包括：

```text
Sentinel-1 GRD:
S1C_IW_GRDH_1SDV_20251228T230337_20251228T230402_005658_00B4C9_5CCC.zip

Sentinel-2 L2A:
S2B_MSIL2A_20251229T035059_N0511_R104_T48RVU_20251229T060507.SAFE.zip
```

两景数据日期相差 1 天，理论上适合做 SAR-Optical 配准实验。但实际检查发现，当前 Sentinel-2 光学数据云量很高，当前 AOI 内 SAR 有效覆盖也不完整，因此这组数据不适合作为最终 baseline 实验数据。

## 2. 项目目录结构

当前项目采用如下结构：

```text
D:/InSAR_Project/
  data/
    raw/
      sentinel1/
      sentinel2/
    processed/
      sar_full_vv_10m.tif
      sar_10m.tif
      optical_10m.tif
  outputs/
    matches_sift.png
    matches_orb.png
    matches_akaze.png
    overlay_before.png
    overlay_after.png
    checkerboard_before.png
    checkerboard_after.png
    metrics.json
    metrics_orb.json
    metrics_akaze.json
    data_quality.json
  scripts/
    01_preprocess_sar.py
    02_preprocess_optical.py
    03_coregister_opencv.py
    04_evaluate_overlay.py
    common.py
  requirements.txt
```

## 3. SAR 预处理流程

SAR 预处理使用 SNAP GPT 完成，脚本为：

```text
scripts/01_preprocess_sar.py
```

处理步骤如下：

1. 读取 Sentinel-1 GRD 产品。
2. 应用轨道文件 `Apply-Orbit-File`。
3. 去除 GRD 边界噪声 `Remove-GRD-Border-Noise`。
4. 辐射定标 `Calibration`，输出 sigma0 强度图。
5. 可选 speckle filtering。
6. 地形校正 `Terrain-Correction`。
7. 输出 GeoTIFF。

当前实际采用的是先生成整景 VV 单极化地形校正结果，再裁剪 AOI：

```powershell
python scripts/01_preprocess_sar.py `
  --input S1C_IW_GRDH_1SDV_20251228T230337_20251228T230402_005658_00B4C9_5CCC.zip `
  --output data/processed/sar_full_vv_10m.tif `
  --polarizations VV `
  --map-projection EPSG:32648 `
  --pixel-spacing 10 `
  --dem-name "Copernicus 30m Global DEM" `
  --gpt "C:\Program Files\esa-snap\bin\gpt.exe"
```

这样做的原因是：整景 VV 单波段输出小于 4GB，能避开经典 GeoTIFF 4GB 限制；如果直接输出整景 VV+VH 双极化，文件可能超过 GeoTIFF 限制。

之后把整景 SAR 裁剪到兴隆湖附近 AOI，得到：

```text
data/processed/sar_10m.tif
```

## 4. Optical 预处理流程

Optical 预处理脚本为：

```text
scripts/02_preprocess_optical.py
```

处理步骤如下：

1. 读取 Sentinel-2 L2A `.SAFE.zip`。
2. 解压到 `data/raw/sentinel2/`。
3. 查找 10 m 波段：
   - `B04`
   - `B03`
   - `B02`
4. 生成 RGB GeoTIFF。
5. 如果提供 SAR reference，则把 Sentinel-2 重采样到 SAR 的 CRS、分辨率、范围和尺寸。

当前命令：

```powershell
python scripts/02_preprocess_optical.py `
  --input S2B_MSIL2A_20251229T035059_N0511_R104_T48RVU_20251229T060507.SAFE.zip `
  --output data/processed/optical_10m.tif `
  --reference data/processed/sar_10m.tif
```

输出结果：

```text
data/processed/optical_10m.tif
```

此时 SAR 和 Optical 已经具有相同的：

```text
CRS
pixel size
width / height
geographic bounds
affine transform
```

## 5. 空间统一

本项目中以 Sentinel-1 SAR 裁剪结果作为参考网格，把 Sentinel-2 光学图像重采样到 SAR 网格。

当前统一后的空间参数：

```text
CRS: EPSG:32648
Resolution: 10 m
Size: 1743 x 1787
```

这种处理保证后续 OpenCV 配准时，两张图像已经处于同一像素网格下。理论上，如果地理编码和元数据正确，两张图像应当已经具有较好的地理对齐关系。

## 6. OpenCV Baseline 配准流程

配准脚本为：

```text
scripts/03_coregister_opencv.py
```

baseline 流程如下：

```text
SAR intensity / Optical RGB
        ↓
灰度化与归一化
        ↓
特征点检测与描述
        ↓
descriptor matching
        ↓
Lowe ratio test
        ↓
RANSAC mismatch removal
        ↓
估计 affine / homography
        ↓
warp moving image
        ↓
输出配准结果和指标
```

当前 SAR 作为 moving image，Optical 作为 reference image：

```powershell
python scripts/03_coregister_opencv.py `
  --moving data/processed/sar_10m.tif `
  --reference data/processed/optical_10m.tif `
  --method sift `
  --model affine `
  --sar-log
```

其中：

```text
--sar-log
```

表示对 SAR 强度图做：

```text
log(1 + intensity)
```

这样可以压缩 SAR 强反射值，改善可视化和特征提取。

测试过的方法包括：

```text
SIFT + affine
ORB + partial-affine
AKAZE + partial-affine
SIFT + homography
```

## 7. 配准结果评价

评价脚本为：

```text
scripts/04_evaluate_overlay.py
```

输出包括：

```text
outputs/overlay_before.png
outputs/overlay_after.png
outputs/checkerboard_before.png
outputs/checkerboard_after.png
```

评价指标包括：

```text
keypoints_moving
keypoints_reference
matches_after_ratio
ransac_inliers
inlier_ratio
inlier_rmse_pixels
transform_matrix
```

这些指标保存在：

```text
outputs/metrics.json
outputs/metrics_orb.json
outputs/metrics_akaze.json
outputs/metrics_sift_homography.json
```

## 8. 当前数据质量检查结论

数据质量报告保存在：

```text
outputs/data_quality.json
```

当前结果：

```text
SAR valid nonzero pixels: 53.97%
Optical valid nonzero pixels: 83.67%
SAR and Optical both valid: 47.07%
SAR valid inside Optical valid: 87.21%
Optical valid inside SAR valid: 56.26%
```

另外，Sentinel-2 的 `SCL_20m` 场景分类显示：

```text
cloud pixels: about 83.66%
nodata pixels: about 16.34%
```

这说明当前 Sentinel-2 图像在 AOI 内基本被云覆盖，无法提供稳定的道路、湖岸线、建筑边界等光学纹理。

同时，SAR 裁剪结果只有约 54% 有效非零像素，说明当前 AOI 有一部分位于 Sentinel-1 地形校正后的无效区域或有效覆盖边缘。

因此，当前 SIFT / ORB / AKAZE 的 RANSAC inlier 数很少，不应被解释为算法实现错误，而应优先解释为数据质量和覆盖问题。

## 9. 当前问题总结

当前数据不适合作为正式 baseline 的原因：

1. Sentinel-2 光学图像云量过高。
2. Optical 有明显 nodata 区域。
3. SAR 有效覆盖不完整。
4. SAR-Optical 有效重叠区域不足。
5. 跨模态纹理差异本身较大，普通 SIFT/ORB 对 SAR-Optical 不够鲁棒。

## 10. 推荐后续流程

下一步建议先重新选择数据，而不是继续调 OpenCV 参数。

推荐数据选择标准：

1. Sentinel-2 L2A：
   - 云量低于 10%。
   - AOI 不在 tile 边缘。
   - 使用 `SCL` 检查 AOI 内是否有大量 cloud / nodata。

2. Sentinel-1 GRD：
   - AOI 尽量位于影像有效覆盖中心区域。
   - 优先选择同轨、近日期产品。
   - 先检查 terrain correction 后的有效像素比例。

3. 重新跑流程：
   - SAR preprocessing。
   - Optical preprocessing。
   - Spatial grid alignment。
   - SIFT / ORB / AKAZE baseline。
   - Metrics and overlay evaluation。

如果换成低云光学图像和有效 SAR 覆盖后，传统方法仍然效果差，再进入后续改进方向：

```text
HOPC
CFOG
RIFT
PCSD
Nonlinear Diffusion + UND-Harris
Siamese Network
Domain Adaptation
```

## 11. 与文献标准流程的对应关系

文献中的标准流程通常是：

```text
feature detection
→ feature description
→ feature matching
→ mismatch removal
→ transformation estimation
→ image resampling
```

当前 baseline 对应为：

```text
SIFT / ORB / AKAZE feature detection and description
→ descriptor matching
→ Lowe ratio test
→ RANSAC
→ affine / homography estimation
→ SAR warping to Optical grid
```

后续可以把 SIFT/ORB/AKAZE 替换为更适合 SAR-Optical 的结构特征方法，例如 CFOG、RIFT、HOPC 或 PCSD。
