#!/usr/bin/env bash
#SBATCH --job-name=utkface_analyze
#SBATCH --output=habrok/logs/utkface_analyze-%j.out
#SBATCH --error=habrok/logs/utkface_analyze-%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=regular
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Run AFTER all 9 array tasks of run_utkface.slurm finish.
# Produces tables, calibration plots, and the qualitative panel under
# code/utkface_results/figures/.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR}"
cd "${REPO_ROOT}"

mkdir -p habrok/logs

module purge
module load Python/3.11.5-GCCcore-13.2.0
source "${REPO_ROOT}/venv/bin/activate"

cd "${REPO_ROOT}/code"
python -u analyze_utkface.py \
    --utkface_dir "${REPO_ROOT}/utkface" \
    --utk_results_dir "utkface_results" \
    --appa_results_dir "mc_dropout_results"

echo "Analysis done."
