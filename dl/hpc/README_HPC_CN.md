# 超算运行说明

本目录只保留完整端到端流水线相关文件。默认流程是：

```text
build index
训练 G2-G7/G9
运行 G8 MAP-Net-like image-based 对比
test split 评估
registration 推理
汇总 summary.json
```

## 1. 上传内容

当前代码和数据都在 `dl/` 中。迁移到超算时上传：

```text
dl/
```

训练数据位于：

```text
dl/data/extracted/
  SAR_1024/
  OPT_1024/
  LAB_1024/
```

当前配准训练主要使用 SAR 和 Optical，Label 暂不作为配准监督标签。

## 2. 环境安装

推荐 conda：

```bash
conda env create -f dl/hpc/environment.yml
conda activate saropt-dl
```

如果集群需要手动安装 PyTorch，请按集群 CUDA 版本安装，然后补齐：

```bash
pip install rasterio opencv-python scikit-learn tqdm optuna
```

## 3. 完整流水线运行

普通 bash 运行：

```bash
bash dl/hpc/run_full_pipeline.sh
```

SLURM 提交：

```bash
sbatch dl/hpc/slurm/full_pipeline.slurm
```

直接用 Python 运行：

```bash
python dl/run_full_pipeline.py --device cuda --amp
```

## 4. 输出目录

完整流水线汇总：

```text
dl/outputs/full_pipeline/<timestamp>/
  pipeline_config.json
  pipeline.log
  summary.json
```

每个模型训练输出：

```text
dl/outputs/runs/<Gx_model>/
  config.json
  metrics.jsonl
  train_history.json
  checkpoints/
    best.pt
    last.pt
```

## 5. 常用参数

通过环境变量控制 bash 包装脚本：

```bash
EPOCHS=30 SAMPLES_PER_EPOCH=80000 BATCH_SIZE=32 NUM_WORKERS=8 bash dl/hpc/run_full_pipeline.sh
```

配准阶段和 G8 参数也可以通过环境变量控制：

```bash
REGISTRATION_GRID_STEP=256 REGISTRATION_SEARCH_RADIUS=16 G8_MAX_SIZE=512 bash dl/hpc/run_full_pipeline.sh
```

如果只想跑部分模型，用 Python 入口：

```bash
python dl/run_full_pipeline.py --models G5,G8,G9 --device cuda --amp
```

跳过配准推理：

```bash
python dl/run_full_pipeline.py --device cuda --amp --skip-registration
```

已有 checkpoint 时跳过对应训练：

```bash
python dl/run_full_pipeline.py --device cuda --amp --skip-existing
```

开启 Bayesian Optimization 后再正式训练：

```bash
USE_OPTUNA=1 bash dl/hpc/run_full_pipeline.sh
```

SLURM 提交时推荐：

```bash
sbatch --export=ALL,USE_OPTUNA=1 dl/hpc/slurm/full_pipeline.slurm
```

手动控制 trial 数：

```bash
USE_OPTUNA=1 OPTUNA_TRIALS=20 bash dl/hpc/run_full_pipeline.sh
```

Python 直接运行：

```bash
python dl/run_full_pipeline.py --device cuda --amp --use-optuna
```

默认 trial 数：

```text
G2-G5: 30 trials
G6-G7: 50 trials
G9: 80 trials
G8: 不做 patch-based Bayesian Optimization
```

## 6. 提交前必须修改

打开：

```text
dl/hpc/slurm/full_pipeline.slurm
```

按超算实际情况修改：

```text
--partition
--account
--gres
--time
module load / conda activate
PROJECT_DIR
```

## 7. 注意事项

1. 数据划分必须使用 `dl/outputs/pairs.csv`，不要按 patch 随机划分。
2. 建议开启 `--amp`，降低显存占用。
3. G7 和 G9 显存占用更高，必要时降低 batch size。
4. G8 不进行 patch 训练，它直接对整图做 image-based registration。
5. 每个模型的结果会写入独立 run 目录，不会相互覆盖。
6. 如果共享文件系统 I/O 较慢，可以把 `dl/data/extracted` 复制到节点本地盘后重新生成索引。
