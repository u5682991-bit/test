# SAR-Optical 深度学习配准阶段进度汇报

## 1. 当前目标

本阶段目标是把项目从传统手工特征配准扩展到深度学习方向，主线任务为：

```text
SAR image + Optical image
-> patch matching
-> 高置信度匹配点筛选
-> RANSAC 删除错误匹配
-> Affine Transform / Homography
-> 最终图像配准
```

当前已将深度学习相关内容集中放入新的 `dl/` 目录，与原有传统方法脚本分离。

## 2. 数据检查结果

深度学习数据位于：

```text
dl/data/extracted/`r`n  SAR_1024/`r`n  OPT_1024/`r`n  LAB_1024/
```

检查结果：

| 数据类型 | 数量 | 通道 | 尺寸 | 数据类型 | 说明 |
|---|---:|---:|---|---|---|
| SAR | 610 | 1 | 1024 x 1024 | uint8 | 可用 |
| Optical | 609 | 3 | 1024 x 1024 | uint8 | 缺少编号 022 |
| Label | 610 | 3 | 1024 x 1024 | uint8 | RGB 语义标签，当前不用于配准训练 |

完整 SAR-Optical 配对数量：

```text
609 对
```

缺失样本：

```text
JiuJiang_01_01_04_01_Optical_022.tif
```

当前结论：

```text
SAR 和 Optical 图像对可以用于深度学习配准训练。
Label 不是位移场、控制点或仿射矩阵，因此不能直接作为配准监督标签。
第一阶段训练暂时忽略 Label。
```

## 3. 数据划分

已实现按原始 image pair 进行划分，而不是按 patch 随机划分。

索引文件：

```text
dl/outputs/pairs.csv
```

当前划分结果：

| Split | Image Pairs |
|---|---:|
| Train | 426 |
| Validation | 91 |
| Test | 92 |
| Total | 609 |

这样可以避免同一张原图切出的 patch 同时出现在 train 和 test 中，防止 data leakage 和虚高结果。

实际流程为：

```text
原始 SAR-Optical image pairs
-> train / val / test split
-> 在线生成 patch pairs
-> 数据增强
-> 模型训练
```

## 4. 已创建的 dl 目录结构

```text
dl/
  README.md
  DEEP_LEARNING_PROGRESS_REPORT_CN.md
  requirements-dl.txt
  configs/
    experiment_groups.json
  outputs/
    pairs.csv
    checkpoints/
  scripts/
    build_index.py
    train_patch_matcher.py
    evaluate_patch_matcher.py
    optimize_hparams.py
    register_pair.py
  src/
    data/
      pairs.py
      dataset.py
    models/
      cbam.py
      matchers.py
      factory.py
    image_based/
      README.md
    utils/
      metrics.py
```

## 5. 已实现的实验组

当前已实现文档中的主要 patch-based 深度学习实验组：

| 实验组 | 模型名称 | 代码 model 参数 | 状态 |
|---|---|---|---|
| G2 | ResNet18 Baseline | `resnet18` | 已实现 |
| G3 | ResNet18 + CBAM | `resnet18_cbam` | 已实现 |
| G4 | Pseudo-Siamese ResNet18 | `ps_resnet18` | 已实现 |
| G5 | Pseudo-Siamese ResNet18 + CBAM | `ps_resnet18_cbam` | 已实现 |
| G6 | Pseudo-Siamese ResNet34 + CBAM | `ps_resnet34_cbam` | 已实现 |
| G7 | Pseudo-Siamese ResNet50 + CBAM | `ps_resnet50_cbam` | 已实现 |
| G8 | MAP-Net-like image-based baseline | `mapnet_like` | 已实现 |
| G9 | Final Multi-scale Model | `final_multiscale` | 已实现 |

G1 传统方法不放入深度学习模型工厂，继续使用原有 `scripts/` 中的传统配准脚本。

G8 MAP-Net 是 image-based 文献对比组，已经在 `dl/src/image_based/` 中独立实现，不与 patch-based 模型混合。

G8 当前实现内容：

```text
Full SAR image + Full Optical image
-> ResNet50 conv4 dense feature map
-> SPAP: 4x4 / 2x2 / 1x1 spatial pyramid average pooling
-> attention saliency 选择高响应特征点
-> PCA 降维
-> KD-tree 最近邻匹配
-> RANSAC
-> affine transform
```

## 6. G9 Final Proposed Model 实现情况

G9 当前实现为：

```text
Patch-based Pseudo-Siamese ResNet34/50
+ CBAM Attention
+ Multi-scale Patch Fusion
```

支持的 backbone：

```text
ResNet34
ResNet50
```

支持的 patch size combination：

```text
64 + 128
128 + 256
64 + 128 + 256
```

支持的 fusion weight：

```text
可手动配置
可通过 Optuna 搜索
```

G9 的基本流程：

```text
SAR patch / Optical patch
-> Pseudo-Siamese branches
-> ResNet34 or ResNet50 feature extraction
-> CBAM attention
-> feature_64 / feature_128 / feature_256
-> multi-scale fusion
-> matching score
```

## 7. 数据增强实现情况

已实现文档中要求的数据增强：

| 增强方式 | 状态 | 说明 |
|---|---|---|
| Random crop | 已实现 | 在线 patch sampling 即随机裁剪 |
| Rotation | 已实现 | 默认 ±10 度，可调到 ±15 度 |
| Scale | 已实现 | 默认 0.9-1.1，可调到 0.8-1.2 |
| Small translation | 已实现 | 默认 ±4 px，可调 |
| Flip | 已实现 | 默认水平翻转概率 0.5 |
| Optical brightness / contrast | 已实现 | 仅作用于 Optical |
| SAR Gaussian noise | 未使用 | 按实验设计要求避免使用 |

训练脚本中可配置参数：

```powershell
--rotation-deg
--scale-min
--scale-max
--translation-px
--flip-prob
--optical-contrast
--optical-brightness
```

## 8. Bayesian Optimization 实现情况

已创建 Optuna 调参脚本：

```text
dl/scripts/optimize_hparams.py
```

目前已接入完整流水线：

```text
python dl/run_full_pipeline.py --device cuda --amp --use-optuna
```

开启后流程为：

```text
对 G2-G7/G9 分别运行 Optuna
-> 读取 best params
-> 使用 best params 正式训练
-> test 评估
-> registration
-> summary 汇总
```

当前支持搜索：

```text
learning rate
batch size
weight decay
dropout
CBAM reduction ratio
attention placement
rotation degree
scale range
translation range
```

针对 G9 额外支持搜索：

```text
final backbone: ResNet34 / ResNet50
patch size combination: 64+128 / 128+256 / 64+128+256
patch fusion weight
```

优化目标：

```text
validation F1-score
```

后续可以扩展为联合考虑：

```text
F1-score
AUC
registration RMSE
NCM
```

## 9. 评价指标实现情况

Patch Matching 指标已实现：

```text
Accuracy
Precision
Recall
F1-score
AUC
```

Registration 指标在 `register_pair.py` 中输出：

```text
NCM: Number of Candidate Matches
CMR: Correct Match Ratio, 当前用 RANSAC inlier ratio 表示
RMSE: RANSAC inlier residual RMSE
Runtime
Parameter Count
```

## 10. 已验证内容

已成功运行数据索引构建：

```powershell
python dl/scripts/build_index.py --data-root dl/data/extracted --out dl/outputs/pairs.csv
```

输出结果：

```text
total_complete_pairs: 609
train: 426
val: 91
test: 92
missing_optical_022_excluded: true
```

已成功运行一次小规模 smoke training：

```powershell
python dl/scripts/train_patch_matcher.py `
  --pairs dl/outputs/pairs.csv `
  --model ps_resnet18_cbam `
  --epochs 1 `
  --samples-per-epoch 32 `
  --val-samples 16 `
  --batch-size 4 `
  --num-workers 0 `
  --device cuda
```

验证通过：

```text
PyTorch 可用
CUDA 可用
数据加载可用
模型前向传播可用
反向传播可用
checkpoint 保存可用
评估脚本可用
```

已运行语法检查：

```powershell
python -m compileall dl
```

结果：

```text
通过
```

## 11. 当前限制

当前阶段仍有以下限制：

1. 还没有进行正式长时间训练。
2. smoke training 只用于验证代码链路，指标没有实际实验意义。
3. 当前 patch matching 的正样本默认认为 SAR-Optical 原始图像对已经近似对齐。
4. Label 当前未用于训练，因为它是 RGB 语义标签，不是配准标签。
5. MAP-Net 目前仅预留 image-based 目录，还没有完整复现。
6. Registration 阶段已经有脚本入口，但需要经过正式训练后的 checkpoint 才能得到有意义结果。

## 11.1 超算运行准备

已补充超算运行相关文件：

```text
dl/run_full_pipeline.py
dl/run_full_pipeline.ps1
dl/hpc/README_HPC_CN.md
dl/hpc/TRANSFER_CHECKLIST_CN.md
dl/hpc/environment.yml
dl/hpc/run_full_pipeline.sh
dl/hpc/slurm/full_pipeline.slurm
```

完整端到端流水线：

```text
dl/run_full_pipeline.py
```

该脚本会一次性执行：

```text
build index
train G2-G7/G9
run G8 MAP-Net-like image-based baseline
evaluate each model
registration inference
write summary.json
```

训练脚本已增强为适合超算长任务运行：

```text
--run-name      独立实验目录
--resume        从 last.pt 断点续训
--amp           CUDA mixed precision
--seed          固定随机种子
--save-every    定期保存 epoch checkpoint
```

每次训练输出目录：

```text
dl/outputs/runs/<run-name>/
  config.json
  metrics.jsonl
  train_history.json
  checkpoints/
    best.pt
    last.pt
```

## 12. 推荐下一步

建议先正式训练 G5：

```powershell
python dl/scripts/train_patch_matcher.py `
  --pairs dl/outputs/pairs.csv `
  --model ps_resnet18_cbam `
  --epochs 20 `
  --samples-per-epoch 50000 `
  --val-samples 4096 `
  --batch-size 32 `
  --device cuda
```

然后依次训练并对比：

```text
G2: resnet18
G3: resnet18_cbam
G4: ps_resnet18
G5: ps_resnet18_cbam
G6: ps_resnet34_cbam
G7: ps_resnet50_cbam
G9: final_multiscale
```

推荐先不直接跑 G7/G9 的大规模训练，先确认 G5 的 patch matching 指标能够稳定提升。

## 13. 推荐实验顺序

建议按以下顺序推进：

```text
1. G2 ResNet18 baseline
2. G3 ResNet18 + CBAM
3. G4 Pseudo-Siamese ResNet18
4. G5 Pseudo-Siamese ResNet18 + CBAM
5. G6 Pseudo-Siamese ResNet34 + CBAM
6. G7 Pseudo-Siamese ResNet50 + CBAM
7. G8 MAP-Net-like image-based 文献对比
8. G9 Multi-scale fusion final model
9. 高置信度匹配点 + RANSAC registration
```

这样可以完整支撑消融实验：

```text
G2 vs G3: CBAM 是否有效
G2 vs G4: Pseudo-Siamese 是否有效
G4 vs G5: CBAM 在双分支中是否有效
G5 vs G6: 更深 backbone 是否有效
G6 vs G7: ResNet50 是否带来提升或过拟合
G5/G6/G7 vs G9: Multi-scale Patch Fusion 是否有效
G8 vs G9: MAP-Net 与最终模型对比
```

## 14. 当前结论

目前深度学习实验框架已经完成第一版工程实现：

```text
数据索引
image-pair 级 train/val/test 划分
在线 patch pair 生成
数据增强
ResNet / CBAM / Pseudo-Siamese / Multi-scale 模型
训练脚本
评估脚本
Optuna 调参脚本
高置信匹配点 + RANSAC 配准脚本
```

因此，项目已经可以进入正式深度学习训练阶段。

