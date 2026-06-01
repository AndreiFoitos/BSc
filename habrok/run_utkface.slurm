#!/usr/bin/env bash
#SBATCH --job-name=utkface_eval
#SBATCH --output=habrok/logs/utkface-%A_%a.out
#SBATCH --error=habrok/logs/utkface-%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-8

# 9 array tasks = 3 methods x 3 fractions (25, 50, 100).
# Each task evaluates one (method, fraction) on the full UTKFace set, 20 MC passes.
# Wall clock per task on an A100: ~30-60 min for 23,705 imgs.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR}"
cd "${REPO_ROOT}"

mkdir -p habrok/logs

module purge
module load Python/3.11.5-GCCcore-13.2.0
source "${REPO_ROOT}/venv/bin/activate"

# Map SLURM_ARRAY_TASK_ID -> (method, fraction)
METHODS=(dropconnect dropconnect dropconnect flipout flipout flipout ensemble ensemble ensemble)
FRACTIONS=(25 50 100 25 50 100 25 50 100)

METHOD="${METHODS[$SLURM_ARRAY_TASK_ID]}"
FRACTION="${FRACTIONS[$SLURM_ARRAY_TASK_ID]}"

echo "Task ${SLURM_ARRAY_TASK_ID}: method=${METHOD} fraction=${FRACTION}"
echo "GPU info:"
nvidia-smi || true
echo

cd "${REPO_ROOT}/code"
python -u predict_utkface.py \
    --utkface_dir "${REPO_ROOT}/utkface" \
    --models_dir "trained_models_by_fraction" \
    --results_dir "utkface_results" \
    --methods "${METHOD}" \
    --fractions "${FRACTION}" \
    --mc_samples 20 \
    --batch_size 64

echo "Task ${SLURM_ARRAY_TASK_ID} done."
