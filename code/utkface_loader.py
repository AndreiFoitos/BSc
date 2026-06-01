"""UTKFace data loader for cross-dataset evaluation.

Filename convention: [age]_[gender]_[race]_[date&time].jpg
We only use [age] (chronological). UTKFace has no per-image apparent-age std,
so the dataset returned here yields (image, real_age) pairs only.

Preprocessing intentionally mirrors predict.py (img/255.0) so UTKFace results
are comparable to the paper's APPA-REAL numbers, which were produced the same
way. NOTE: training used densenet.preprocess_input — that mismatch is a
separate issue documented in the project notes.
"""

import os
import re
import numpy as np
import pandas as pd
import tensorflow as tf


_FNAME_RE = re.compile(r"^(\d+)_(\d+)_(\d+)_\d+\.jpg(?:\.chip\.jpg)?$")


def parse_filename(fname):
    """Return (age, gender, race) or None if the name doesn't match."""
    m = _FNAME_RE.match(fname)
    if not m:
        return None
    age, gender, race = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if age < 0 or age > 120:
        return None
    return age, gender, race


def index_directory(data_dir):
    """Scan data_dir for UTKFace jpgs; return a DataFrame with file_name, path, age, gender, race."""
    rows = []
    for fname in os.listdir(data_dir):
        parsed = parse_filename(fname)
        if parsed is None:
            continue
        age, gender, race = parsed
        rows.append({
            "file_name": fname,
            "path": os.path.join(data_dir, fname),
            "age": age,
            "gender": gender,
            "race": race,
        })
    df = pd.DataFrame(rows).sort_values("file_name").reset_index(drop=True)
    return df


def load_utkface_dataset(data_dir, img_size=(224, 224), batch_size=32, limit=None,
                         stratify_bins=10, seed=42):
    """Load UTKFace images and ages into a tf.data.Dataset of (image, age).

    Args:
        data_dir: folder containing aligned & cropped UTKFace jpgs.
        img_size: target (H, W) — must match training (224, 224).
        batch_size: inference batch size.
        limit: optional int; when set, subsample to N images stratified by age
            decade. If limit >= total, returns the full set.
        stratify_bins: how many age bins to stratify across when limit is set.
        seed: random seed for reproducible sub-sampling.

    Returns:
        (dataset, df) where df is the index used (so callers can join back to
        ages, file names, demographics for qualitative panels).
    """
    df = index_directory(data_dir)
    if limit is not None and limit < len(df):
        df["_age_bin"] = pd.cut(df["age"], bins=stratify_bins, labels=False)
        per_bin = max(1, limit // stratify_bins)
        rng = np.random.default_rng(seed)
        parts = []
        for b, sub in df.groupby("_age_bin"):
            take = min(per_bin, len(sub))
            idx = rng.choice(len(sub), size=take, replace=False)
            parts.append(sub.iloc[idx])
        df = pd.concat(parts).drop(columns="_age_bin").sort_values("file_name").reset_index(drop=True)

    if len(df) == 0:
        raise RuntimeError(f"No UTKFace-formatted images found in {data_dir}")

    paths = df["path"].tolist()
    ages = df["age"].astype(np.float32).values

    def _load(path):
        img = tf.keras.utils.load_img(path, target_size=img_size)
        arr = tf.keras.utils.img_to_array(img) / 255.0
        return arr.astype(np.float32)

    images = np.stack([_load(p) for p in paths], axis=0)
    labels = {"real_age": ages}

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.batch(batch_size, drop_remainder=False)
    return dataset, df


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "../utkface"
    df = index_directory(data_dir)
    print(f"Indexed {len(df)} UTKFace images from {data_dir}")
    if len(df) > 0:
        print(df.head())
        print("Age range:", df["age"].min(), "-", df["age"].max())
        print("Age distribution (decade bins):")
        print(pd.cut(df["age"], bins=range(0, 121, 10)).value_counts().sort_index())
