# GitHub 上传与运行检查

## 1. 结论

当前 `dl/` 已经整理为可上传 GitHub 的运行包：

```text
代码、数据、配置、超算脚本都在 dl/ 内
完整入口为 dl/run_full_pipeline.py
超算 SLURM 入口为 dl/hpc/slurm/full_pipeline.slurm
```

## 2. 数据体积

训练数据位于：

```text
dl/data/extracted/
```

当前数据规模：

```text
SAR_1024: 610
OPT_1024: 609
LAB_1024: 610
总数据量约 3.42 GB
单个文件未超过 100 MB
```

因为总数据量较大，数据文件应通过 Git LFS 管理。

## 3. Git LFS

已添加：

```text
.gitattributes
```

其中配置：

```text
dl/data/extracted/**/*.tif filter=lfs diff=lfs merge=lfs -text
```

上传前需要在本机执行：

```bash
git lfs install
git lfs track "dl/data/extracted/**/*.tif"
```

然后正常提交：

```bash
git add .gitattributes .gitignore dl/
git commit -m "Add SAR-Optical deep learning pipeline"
git push
```

## 4. 组员克隆后运行

组员需要先安装 Git LFS：

```bash
git lfs install
git clone <repo-url>
cd <repo>
git lfs pull
```

然后创建环境：

```bash
conda env create -f dl/hpc/environment.yml
conda activate saropt-dl
```

生成索引检查数据：

```bash
python dl/scripts/build_index.py \
  --data-root dl/data/extracted \
  --out dl/outputs/pairs.csv
```

预期输出：

```text
total_complete_pairs: 609
missing_optical_022_excluded: true
```

## 5. 运行完整实验

固定参数完整实验：

```bash
python dl/run_full_pipeline.py --device cuda --amp
```

贝叶斯优化 + 正式训练：

```bash
python dl/run_full_pipeline.py --device cuda --amp --use-optuna
```

超算 SLURM 固定参数：

```bash
sbatch dl/hpc/slurm/full_pipeline.slurm
```

超算 SLURM 贝叶斯优化：

```bash
sbatch --export=ALL,USE_OPTUNA=1 dl/hpc/slurm/full_pipeline.slurm
```

## 6. 输出目录

完整流水线输出：

```text
dl/outputs/full_pipeline/<timestamp>/summary.json
```

每个模型输出：

```text
dl/outputs/runs/<Gx_model>/
```

这些输出目录已写入 `.gitignore`，不应提交到 GitHub。

## 7. 上传前确认

上传前建议检查：

```bash
git status
git lfs ls-files
```

确认 `.tif` 文件被 Git LFS 跟踪，而不是作为普通 Git blob 提交。

