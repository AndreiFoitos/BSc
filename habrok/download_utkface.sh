#!/usr/bin/env bash
# Download UTKFace aligned & cropped on Habrok via Hugging Face.
# Faster than uploading 23,708 small files through the OnDemand portal.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${REPO_ROOT}/utkface"

if [[ -d "${TARGET_DIR}" ]] && [[ "$(ls -A "${TARGET_DIR}" 2>/dev/null | wc -l)" -gt 1000 ]]; then
    echo "UTKFace appears to already be present at ${TARGET_DIR}"
    echo "Skipping download. Delete the folder first if you want to re-download."
    exit 0
fi

mkdir -p "${TARGET_DIR}"

module purge
module load Python/3.11.5-GCCcore-13.2.0

# Use a temp pip install — keeps the eval venv clean
python -m pip install --user --quiet huggingface_hub

python - <<'PY'
import os, sys
from huggingface_hub import snapshot_download

target = os.environ.get("TARGET_DIR")
print(f"Downloading nu-delta/utkface to {target} ...")
path = snapshot_download(
    repo_id="nu-delta/utkface",
    repo_type="dataset",
    local_dir=target,
    local_dir_use_symlinks=False,
)
print(f"Snapshot at {path}")

# Flatten nested folders if the HF repo wraps files in subdirs
import shutil
moved = 0
for root, _, files in os.walk(target):
    if root == target:
        continue
    for fname in files:
        if fname.endswith(".jpg") or fname.endswith(".jpg.chip.jpg"):
            src = os.path.join(root, fname)
            dst = os.path.join(target, fname)
            if not os.path.exists(dst):
                shutil.move(src, dst)
                moved += 1
print(f"Flattened {moved} files into {target}")

# Quick sanity check
n = sum(1 for f in os.listdir(target) if f.endswith(".jpg"))
print(f"Total .jpg files under {target}: {n}")
PY

echo "UTKFace download done."
