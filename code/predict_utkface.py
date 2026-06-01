"""Run MC inference of trained APPA-REAL models on UTKFace (cross-dataset eval).

Default: evaluate 25%, 50%, 100% data fractions for DropConnect, Flipout,
Ensemble. Outputs one CSV per (method, fraction) under utkface_results/ with
columns (y_true, mean_prediction, aleatoric_uncertainty, epistemic_uncertainty,
predictive_uncertainty, model_predicted_std) — identical schema to predict.py
so the existing analysis utilities are reusable.

Usage:
    python predict_utkface.py --utkface_dir ../utkface
    python predict_utkface.py --utkface_dir ../utkface --fractions 100
    python predict_utkface.py --utkface_dir ../utkface --smoke
"""

import argparse
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

# Importing model.py registers AgeEstimationModel + DropConnectDense via the
# @register_keras_serializable decorator. Required for load_model to deserialize.
import model  # noqa: F401
from utkface_loader import load_utkface_dataset

MODELS_DIR = "trained_models_by_fraction"
DEFAULT_RESULTS_DIR = "utkface_results"
DEFAULT_FRACTIONS = (25, 50, 100)
MC_SAMPLES = 20
BATCH_SIZE = 32


def mc_inference(models, dataset, n_samples, label_key="real_age"):
    """One-pass-per-sample MC inference adapted from predict.py.

    `models` is a list — for DropConnect/Flipout it's [single_model] and the
    stochasticity comes from training=True; for Ensembles it's the 3 ensemble
    members and we additionally apply training=True on each (cheap insurance,
    matches what predict.py does for ensembles).

    Memory-conscious layout: we iterate the dataset ONCE and run all n_samples
    MC passes per batch. Each image is therefore read from disk a single time
    (the previous version re-loaded every image n_samples times and held the
    full dataset in RAM, which OOM'd on 32 GB nodes for 23k UTKFace images).
    Per-image statistics are reduced on the fly so peak memory is bounded by
    O(n_examples + batch_size * n_samples).
    """
    pred_mean_chunks = []
    aleatoric_chunks = []
    epistemic_chunks = []
    pred_std_chunks = []
    y_true_chunks = []

    # tqdm without total — it counts as we iterate. We deliberately avoid a
    # dry pass to compute n_batches because that would trigger 23k JPEG
    # decodes for no benefit.
    pbar = tqdm(desc="Batches", unit="batch")
    for images, labels in dataset:
        # For each batch, collect n_samples * len(models) predictions.
        batch_means_per_pass = []  # (n_samples, batch_size)
        batch_vars_per_pass = []
        batch_stds_per_pass = []

        for _ in range(n_samples):
            model_means, model_vars, model_stds = [], [], []
            for model in models:
                preds = model(images, training=True)
                mean = preds["apparent_age_avg"].numpy().flatten()
                std = preds["apparent_age_std"].numpy().flatten()
                model_means.append(mean)
                model_vars.append(np.square(std))
                model_stds.append(std)
            batch_means_per_pass.append(np.mean(model_means, axis=0))
            batch_vars_per_pass.append(np.mean(model_vars, axis=0))
            batch_stds_per_pass.append(np.mean(model_stds, axis=0))

        bmp = np.stack(batch_means_per_pass, axis=0)  # (n_samples, batch_size)
        bvp = np.stack(batch_vars_per_pass, axis=0)
        bsp = np.stack(batch_stds_per_pass, axis=0)

        pred_mean_chunks.append(bmp.mean(axis=0))   # (batch_size,)
        aleatoric_chunks.append(bvp.mean(axis=0))
        epistemic_chunks.append(bmp.var(axis=0))
        pred_std_chunks.append(bsp.mean(axis=0))
        y_true_chunks.append(labels[label_key].numpy().flatten())
        pbar.update(1)
    pbar.close()

    pred_mean = np.concatenate(pred_mean_chunks)
    aleatoric = np.concatenate(aleatoric_chunks)
    epistemic = np.concatenate(epistemic_chunks)
    pred_model_std = np.concatenate(pred_std_chunks)
    predictive = aleatoric + epistemic
    y_trues = np.concatenate(y_true_chunks)

    return pred_mean, aleatoric, epistemic, predictive, pred_model_std, y_trues


def collect_model_files(models_dir, fractions, methods):
    """Group .keras files by method (dropconnect/flipout/ensemble) × fraction."""
    by_method_frac = defaultdict(list)
    methods = set(methods)
    for fname in sorted(os.listdir(models_dir)):
        if not fname.endswith(".keras"):
            continue
        ensemble_match = re.match(r"(ensemble)(\d+)percent_model\d+\.keras", fname)
        single_match = re.match(r"(dropconnect|flipout)(\d+)percent\.keras", fname)
        if ensemble_match:
            method, pct = ensemble_match.group(1), int(ensemble_match.group(2))
        elif single_match:
            method, pct = single_match.group(1), int(single_match.group(2))
        else:
            continue
        if pct in fractions and method in methods:
            by_method_frac[(method, pct)].append(fname)
    return by_method_frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--utkface_dir", required=True, help="Folder with aligned/cropped UTKFace jpgs")
    ap.add_argument("--results_dir", default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--models_dir", default=MODELS_DIR)
    ap.add_argument(
        "--fractions", type=int, nargs="+", default=list(DEFAULT_FRACTIONS),
        help="Data fractions to evaluate (must match available checkpoints)",
    )
    ap.add_argument(
        "--methods", type=str, nargs="+", default=["dropconnect", "flipout", "ensemble"],
        choices=["dropconnect", "flipout", "ensemble"],
        help="Restrict to a subset of methods (useful for SLURM array jobs)",
    )
    ap.add_argument("--mc_samples", type=int, default=MC_SAMPLES)
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    ap.add_argument("--max_images", type=int, default=None,
                    help="Stratified-by-age cap on UTKFace images (None = all ~23k)")
    ap.add_argument("--smoke", action="store_true", help="Limit to 64 UTKFace images for a smoke test")
    args = ap.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    print(f"Indexing UTKFace from {args.utkface_dir} ...")
    if args.smoke:
        limit = 64
    elif args.max_images is not None:
        limit = args.max_images
    else:
        limit = None
    dataset, index_df = load_utkface_dataset(
        args.utkface_dir, batch_size=args.batch_size, limit=limit
    )
    n = len(index_df)
    print(f"Loaded {n} images")
    index_df.to_csv(os.path.join(args.results_dir, "utkface_index.csv"), index=False)

    grouped = collect_model_files(args.models_dir, set(args.fractions), args.methods)
    if not grouped:
        raise RuntimeError(
            f"No matching checkpoints under {args.models_dir} for "
            f"fractions={args.fractions} methods={args.methods}"
        )

    for (method, pct), files in sorted(grouped.items()):
        print(f"\n=== {method} @ {pct}% data ({len(files)} model file(s)) ===")
        models = [
            tf.keras.models.load_model(os.path.join(args.models_dir, f), compile=False)
            for f in files
        ]
        pred_mean, aleatoric, epistemic, predictive, pred_model_std, y_true = mc_inference(
            models, dataset, n_samples=args.mc_samples
        )
        out = pd.DataFrame({
            "y_true": y_true,
            "mean_prediction": pred_mean,
            "aleatoric_uncertainty": aleatoric,
            "epistemic_uncertainty": epistemic,
            "predictive_uncertainty": predictive,
            "model_predicted_std": pred_model_std,
        })
        # Attach metadata from index so the analysis step can do demographic slicing
        out = pd.concat([index_df.reset_index(drop=True), out.reset_index(drop=True)], axis=1)
        save_name = f"{method}{pct}percent_utkface_predictions.csv"
        save_path = os.path.join(args.results_dir, save_name)
        out.to_csv(save_path, index=False)

        mae = np.mean(np.abs(out["y_true"] - out["mean_prediction"]))
        rmse = float(np.sqrt(np.mean((out["y_true"] - out["mean_prediction"]) ** 2)))
        print(f"  saved -> {save_path}")
        print(f"  MAE = {mae:.3f}  RMSE = {rmse:.3f}  mean sigma = {out['model_predicted_std'].mean():.3f}")


if __name__ == "__main__":
    main()
