#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

ARGS=(
  dl/run_full_pipeline.py
  --device cuda
  --amp
  --epochs "${EPOCHS:-20}"
  --samples-per-epoch "${SAMPLES_PER_EPOCH:-50000}"
  --val-samples "${VAL_SAMPLES:-4096}"
  --batch-size "${BATCH_SIZE:-32}"
  --num-workers "${NUM_WORKERS:-8}"
  --eval-samples "${EVAL_SAMPLES:-8192}"
  --eval-batch-size "${EVAL_BATCH_SIZE:-64}"
  --registration-sample-ids "${REGISTRATION_SAMPLE_IDS:-100,300,500,610}"
  --registration-grid-step "${REGISTRATION_GRID_STEP:-128}"
  --registration-search-radius "${REGISTRATION_SEARCH_RADIUS:-32}"
  --g8-max-size "${G8_MAX_SIZE:-512}"
  --g8-max-points "${G8_MAX_POINTS:-600}"
  --g8-pca-dim "${G8_PCA_DIM:-64}"
  --g8-ratio "${G8_RATIO:-0.95}"
  --g8-ransac-thresh "${G8_RANSAC_THRESH:-5.0}"
  --g10-region-size "${G10_REGION_SIZE:-256}"
  --g10-block-size "${G10_BLOCK_SIZE:-128}"
  --g10-region-step "${G10_REGION_STEP:-256}"
  --g10-block-step "${G10_BLOCK_STEP:-128}"
  --g10-search-radius "${G10_SEARCH_RADIUS:-192}"
  --g10-threshold "${G10_THRESHOLD:-0.55}"
  --g10-top-k "${G10_TOP_K:-4}"
  --g10-temperature "${G10_TEMPERATURE:-0.05}"
  --g10-batch-size "${G10_BATCH_SIZE:-64}"
)

if [[ -n "${SKIP_REGISTRATION:-}" ]]; then ARGS+=(--skip-registration); fi
if [[ -n "${SKIP_EXISTING:-}" ]]; then ARGS+=(--skip-existing); fi
if [[ -n "${USE_OPTUNA:-}" ]]; then ARGS+=(--use-optuna); fi
if [[ -n "${OPTUNA_TRIALS:-}" ]]; then ARGS+=(--optuna-trials "$OPTUNA_TRIALS"); fi
if [[ -n "${OPTUNA_EPOCHS_PER_TRIAL:-}" ]]; then ARGS+=(--optuna-epochs-per-trial "$OPTUNA_EPOCHS_PER_TRIAL"); fi
if [[ -n "${OPTUNA_SAMPLES_PER_TRIAL:-}" ]]; then ARGS+=(--optuna-samples-per-trial "$OPTUNA_SAMPLES_PER_TRIAL"); fi
if [[ -n "${OPTUNA_VAL_SAMPLES:-}" ]]; then ARGS+=(--optuna-val-samples "$OPTUNA_VAL_SAMPLES"); fi

python "${ARGS[@]}"
