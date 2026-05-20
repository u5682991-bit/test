# SAR-Optical Deep Learning Registration

This folder contains the deep-learning pipeline for SAR-Optical patch matching
and registration. It is intentionally isolated from the handcrafted baseline
scripts in the project root.

## Data

Expected extracted data:

```text
dl/data/extracted/
  SAR_1024/
  OPT_1024/
  LAB_1024/   # optional, not used for registration training
```

The current dataset has 609 complete SAR-Optical pairs. Optical pair `022` is
missing and is excluded automatically.

## Experiment Groups

Implemented model names:

```text
resnet18              # G2
resnet18_cbam         # G3
ps_resnet18           # G4
ps_resnet18_cbam      # G5
ps_resnet34_cbam      # G6
ps_resnet50_cbam      # G7
final_multiscale      # G9, pseudo-siamese + CBAM + multi-scale fusion
g10_hspm_dabm         # G10, HSPM + shared encoder + DABM + correlation matching
```

G1 traditional baselines remain in `scripts/`. MAP-Net is left as a literature
comparison group in `dl/src/image_based/` and is executed as `mapnet_like`.
It is an image-based registration baseline, not a patch-trained model.

## Alignment With Experiment Design

The implemented structure follows the experiment design:

```text
G1: existing handcrafted scripts in ../scripts
G2-G7: patch-based matching models
G8: image-based MAP-Net-like comparison
G9: patch-based pseudo-siamese CBAM model with multi-scale patch fusion
G10: hierarchical sub-block partition + shared dual-branch encoder + DABM + correlation + weighted affine
```

G9 exposes the requested design choices:

```text
backbone: ResNet34 or ResNet50
patch combinations: 64+128, 128+256, 64+128+256
fusion weights: configurable, and searchable by Optuna
```

G10 implements the newer high-precision registration design:

```text
SAR/Optical image
-> hierarchical region/block partition
-> shared-weight dual-branch ResNet encoder
-> CBAM-style channel + spatial attention
-> explicit SAR-vs-Optical correlation matrix
-> softmax weighted sub-pixel coordinate refinement
-> confidence-weighted affine estimation
```

Training still uses the existing patch matching supervision. Registration uses
`dl/scripts/register_g10_hspm.py`, which consumes the trained G10 checkpoint and
performs the correlation-based matching stage.

The data split is image-pair first:

```text
original SAR-Optical pairs -> train / val / test -> online patch generation
```

This is equivalent to the intended anti-leakage rule. It prevents patches from
the same original image pair appearing in both training and testing.

## Quick Start

Run the complete end-to-end pipeline:

```powershell
python dl/run_full_pipeline.py --device cuda --amp
```

Run Bayesian Optimization before final training:

```powershell
python dl/run_full_pipeline.py --device cuda --amp --use-optuna
```

This runs:

```text
build index
train G2-G7/G9
train G10
optional Optuna Bayesian Optimization before final training
run G8 MAP-Net-like image-based registration
evaluate each model on test split
run registration on selected samples
write dl/outputs/full_pipeline/<timestamp>/summary.json
```

Build the pair index and image-level splits:

```powershell
python dl/scripts/build_index.py --data-root dl/data/extracted --out dl/outputs/pairs.csv
```

Run a small smoke training job:

```powershell
python dl/scripts/train_patch_matcher.py --pairs dl/outputs/pairs.csv --model ps_resnet18_cbam --epochs 1 --samples-per-epoch 512 --batch-size 8 --num-workers 0 --run-name smoke_manual
```

Evaluate a checkpoint:

```powershell
python dl/scripts/evaluate_patch_matcher.py --pairs dl/outputs/pairs.csv --checkpoint dl/outputs/runs/smoke_manual/checkpoints/best.pt --model ps_resnet18_cbam
```

Estimate registration for one SAR-Optical pair:

```powershell
python dl/scripts/register_pair.py --pairs dl/outputs/pairs.csv --checkpoint dl/outputs/runs/smoke_manual/checkpoints/best.pt --model ps_resnet18_cbam --sample-id 100
```

Estimate registration with G10:

```powershell
python dl/scripts/register_g10_hspm.py --pairs dl/outputs/pairs.csv --checkpoint dl/outputs/runs/G10_g10_hspm_dabm/checkpoints/best.pt --model g10_hspm_dabm --sample-id 100
```

## Training Logic

The dataset is split by original image pair first. Patches are generated only
after the split, so patches from one original image cannot appear in both train
and test.

The dataset is treated as approximately aligned SAR-Optical positive pairs. For
patch matching, training samples are generated online:

- positive sample: SAR patch and Optical patch from the same image coordinate
- negative sample: SAR patch and Optical patch from different coordinates
- optional augmentation: small rotation, scale, translation, flips, optical
  brightness/contrast

The RGB labels in `LAB_1024` are semantic masks, not registration labels. They
are not used in the first-stage registration pipeline.

Default augmentation follows the experiment design:

```text
SAR geometry: rotation +/-10 deg, scale 0.9-1.1, translation +/-4 px
SAR noise: no Gaussian noise is added
Both modalities: horizontal flip
Optical only: brightness/contrast jitter
Random crop: implicit in online patch sampling
```

You can adjust it from the training command:

```powershell
python dl/scripts/train_patch_matcher.py `
  --rotation-deg 15 `
  --scale-min 0.8 `
  --scale-max 1.2 `
  --translation-px 8 `
  --optical-contrast 0.15 `
  --optical-brightness 12
```

## MAP-Net Boundary

G8 MAP-Net-like is an image-based literature comparison group:

```text
full SAR image + full Optical image
dense feature map
SPAP / attention / PCA / KD-tree
RANSAC registration
```

It is intentionally not included in the patch-based model names. Add it later
to the patch model factory. It is run through `dl/scripts/run_mapnet_g8.py` and
included in `run_full_pipeline.py` summary as an image-based baseline.

## Metrics

Patch matching:

```text
Accuracy, Precision, Recall, F1, AUC
```

Registration:

```text
NCM, CMR, RMSE, runtime, parameter count
```
