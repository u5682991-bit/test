# 超算迁移检查清单

## 必传内容

当前已经把代码和数据都整理到 `dl/` 目录中。迁移到超算时，直接上传整个目录：

```text
dl/
```

其中训练数据位于：

```text
dl/data/extracted/
  SAR_1024/
  OPT_1024/
  LAB_1024/
```

当前配准训练主要使用 `SAR_1024` 和 `OPT_1024`。`LAB_1024` 暂不参与配准训练，但保留在目录中，方便后续扩展。

## 不建议上传的运行产物

如果本地已经产生以下目录，可以不上传：

```text
dl/outputs/runs/
dl/outputs/full_pipeline/
dl/**/__pycache__/
```

这些是训练输出或 Python 缓存，超算上可以重新生成。

## 到超算后第一步

生成数据索引：

```bash
python dl/scripts/build_index.py \
  --data-root dl/data/extracted \
  --out dl/outputs/pairs.csv
```

确认输出包含：

```text
total_complete_pairs: 609
missing_optical_022_excluded: true
```

## 第二步 smoke test

可先跑一个极小训练任务，确认环境、CUDA、数据读取正常：

```bash
python dl/scripts/train_patch_matcher.py \
  --pairs dl/outputs/pairs.csv \
  --model ps_resnet18_cbam \
  --epochs 1 \
  --samples-per-epoch 64 \
  --val-samples 32 \
  --batch-size 4 \
  --num-workers 0 \
  --device cuda \
  --amp \
  --run-name smoke_test
```

## 第三步正式运行完整流水线

使用 SLURM：

```bash
sbatch dl/hpc/slurm/full_pipeline.slurm
```

不用 SLURM 时：

```bash
bash dl/hpc/run_full_pipeline.sh
```

完整流水线包含：

```text
G2-G7/G9 patch-based 模型训练与评估
G8 MAP-Net-like image-based 对比
registration 推理
summary.json 汇总
```

如果要严格按实验设计先做 Bayesian Optimization，再正式训练：

```bash
sbatch --export=ALL,USE_OPTUNA=1 dl/hpc/slurm/full_pipeline.slurm
```

如果想先少量 trial 测试：

```bash
sbatch --export=ALL,USE_OPTUNA=1,OPTUNA_TRIALS=5 dl/hpc/slurm/full_pipeline.slurm
```
