#!/bin/bash
#SBATCH --job-name=saropt-full
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=dl/outputs/slurm/%x-%j.out
#SBATCH --error=dl/outputs/slurm/%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
CONDA_ENV="${CONDA_ENV:-saropt-dl}"

cd "$PROJECT_DIR"
mkdir -p dl/outputs/slurm

# Edit for your cluster.
# module load cuda/12.1
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate "$CONDA_ENV"

bash dl/hpc/run_full_pipeline.sh
