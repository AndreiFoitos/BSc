#!/usr/bin/env bash
# One-time venv setup on Habrok.
# Run on a login node (NOT inside an sbatch job — pip needs internet).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${REPO_ROOT}/venv"

# Load a recent Python with full CPython + headers needed for pip wheels.
module purge
module load Python/3.11.5-GCCcore-13.2.0

if [[ -d "${VENV_DIR}" ]]; then
    echo "venv already exists at ${VENV_DIR} — skipping creation."
else
    python -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

pip install --upgrade pip
pip install -r "$(dirname "${BASH_SOURCE[0]}")/requirements.txt"

echo
echo "Venv ready at ${VENV_DIR}"
echo "Sanity check:"
python -c "import tensorflow as tf; print('TF', tf.__version__); print('GPUs visible to TF (login node may show none):', tf.config.list_physical_devices('GPU'))"
