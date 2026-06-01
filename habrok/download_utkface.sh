#!/usr/bin/env bash
# Download UTKFace aligned & cropped on Habrok via Hugging Face.
# nu-delta/utkface ships as parquet shards (images embedded as bytes),
# so we use the `datasets` library to extract them into individual jpgs
# named [age]_[gender]_[race]_[idx].jpg.chip.jpg to match the original
# UTKFace naming convention our loader expects.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TARGET_DIR="${REPO_ROOT}/utkface"

if [[ -d "${TARGET_DIR}" ]] && [[ "$(find "${TARGET_DIR}" -maxdepth 1 -name '*.jpg' | wc -l)" -gt 1000 ]]; then
    echo "UTKFace appears to already be present at ${TARGET_DIR}"
    echo "Skipping. Remove the folder if you want a fresh extraction."
    exit 0
fi

mkdir -p "${TARGET_DIR}"

module purge
module load Python/3.11.5-GCCcore-13.2.0

# Add user-pip bin to PATH so any prior --user installs are findable
export PATH="$HOME/.local/bin:$PATH"

# huggingface_hub for the download, datasets for parquet+image decoding, Pillow for saving
python -m pip install --user --quiet huggingface_hub datasets pillow pyarrow

python -u - <<'PY'
import os, sys, io
from datasets import load_dataset

target = os.environ["TARGET_DIR"]
os.makedirs(target, exist_ok=True)
print(f"Loading nu-delta/utkface into memory (parquet shards), saving to {target} ...")

ds = load_dataset("nu-delta/utkface", split="train")
print(f"Schema: {ds.features}")
print(f"Rows: {len(ds)}")

# Auto-detect column names (defensive — HF dataset cards vary)
cols = list(ds.column_names)
age_col   = next((c for c in cols if c.lower() in ("age",)), None)
img_col   = next((c for c in cols if c.lower() in ("image", "img", "picture")), None)
gend_col  = next((c for c in cols if c.lower() in ("gender", "sex")), None)
race_col  = next((c for c in cols if c.lower() in ("race", "ethnicity")), None)
print(f"Using columns: age={age_col}, image={img_col}, gender={gend_col}, race={race_col}")
if age_col is None or img_col is None:
    sys.exit(f"ERROR: could not find age/image columns. Available: {cols}")

saved = 0
skipped = 0
for i, row in enumerate(ds):
    try:
        age = int(row[age_col])
        gender = int(row[gend_col]) if gend_col else 0
        race = int(row[race_col]) if race_col else 0
    except (TypeError, ValueError):
        skipped += 1
        continue
    img = row[img_col]
    # img is a PIL.Image after datasets decoding
    fname = f"{age}_{gender}_{race}_{i:08d}.jpg.chip.jpg"
    out_path = os.path.join(target, fname)
    if not os.path.exists(out_path):
        img.convert("RGB").save(out_path, "JPEG", quality=95)
    saved += 1
    if (i + 1) % 2000 == 0:
        print(f"  {i+1}/{len(ds)} processed ({saved} saved, {skipped} skipped)")

print(f"Done. Saved {saved} jpgs, skipped {skipped}.")
n_on_disk = sum(1 for f in os.listdir(target) if f.endswith(".jpg") or f.endswith(".jpg.chip.jpg"))
print(f"Total files on disk under {target}: {n_on_disk}")
PY

echo "UTKFace extraction complete."
