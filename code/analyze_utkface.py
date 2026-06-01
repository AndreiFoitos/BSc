"""Cross-dataset (APPA-REAL vs UTKFace) analysis: tables, calibration, qualitative.

Produces, under utkface_results/figures/:
  - cross_dataset_metrics.csv      : MAE/RMSE/R² per (method, fraction) on each dataset
  - aleatoric_epistemic_compare.png/.csv : avg aleatoric/epistemic on APPA vs UTKFace
  - calibration_<method>_<pct>.png : binned predicted σ vs empirical |error| (UTKFace)
  - calibration_overlay.png        : all methods on one calibration plot @ 100% data
  - ece_uce.csv                    : Expected Calibration Error & UCE per setting
  - qualitative_panel.png          : 6 UTKFace faces — low σ correct, high σ wrong, etc.
"""

import argparse
import os
import re
from glob import glob

import matplotlib
matplotlib.use("Agg")  # headless backend — works on Habrok compute nodes without X
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

UTKFACE_RESULTS_DIR = "utkface_results"
APPA_RESULTS_DIR = "mc_dropout_results"
FIG_DIR_NAME = "figures"


# ---------- helpers ----------

def _basic_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
    }


def _expected_calibration_error(sigma, abs_err, n_bins=10):
    """ECE: weighted average |bin_mean_sigma - bin_mean_error|. Lower is better."""
    edges = np.quantile(sigma, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-9
    idx = np.digitize(sigma, edges) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    ece = 0.0
    total = len(sigma)
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        ece += (m.sum() / total) * abs(sigma[m].mean() - abs_err[m].mean())
    return ece


def _bin_for_calibration(sigma, abs_err, n_bins=10):
    edges = np.quantile(sigma, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-9
    idx = np.digitize(sigma, edges) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        rows.append({
            "bin": b,
            "n": int(m.sum()),
            "sigma_mean": float(sigma[m].mean()),
            "abs_err_mean": float(abs_err[m].mean()),
        })
    return pd.DataFrame(rows)


def _parse_csv_name(fname, suffix):
    """e.g. dropconnect75percent + _mc_predictions.csv -> ('dropconnect', 75)."""
    base = fname.replace(suffix, "")
    m = re.match(r"(dropconnect|flipout|ensemble)(\d+)percent$", base)
    if not m:
        return None
    return m.group(1), int(m.group(2))


# ---------- per-dataset readers ----------

def _read_results(results_dir, suffix):
    out = {}
    for path in sorted(glob(os.path.join(results_dir, f"*{suffix}"))):
        parsed = _parse_csv_name(os.path.basename(path), suffix)
        if parsed is None:
            continue
        method, pct = parsed
        df = pd.read_csv(path)
        out[(method, pct)] = df
    return out


# ---------- analyses ----------

def build_metrics_table(appa_results, utk_results, out_csv):
    rows = []
    for (method, pct), utk_df in sorted(utk_results.items()):
        utk_m = _basic_metrics(utk_df["y_true"], utk_df["mean_prediction"])
        appa_df = appa_results.get((method, pct))
        appa_m = _basic_metrics(appa_df["y_true"], appa_df["mean_prediction"]) if appa_df is not None else {k: np.nan for k in ("MAE", "RMSE", "R2")}
        rows.append({
            "method": method,
            "fraction": pct,
            "APPA_MAE": appa_m["MAE"], "APPA_RMSE": appa_m["RMSE"], "APPA_R2": appa_m["R2"],
            "UTK_MAE": utk_m["MAE"], "UTK_RMSE": utk_m["RMSE"], "UTK_R2": utk_m["R2"],
            "delta_MAE": utk_m["MAE"] - appa_m["MAE"],
        })
    df = pd.DataFrame(rows).sort_values(["method", "fraction"])
    df.to_csv(out_csv, index=False)
    return df


def aleatoric_epistemic_compare(appa_results, utk_results, out_csv, out_png):
    rows = []
    for (method, pct), utk_df in sorted(utk_results.items()):
        appa_df = appa_results.get((method, pct))
        rows.append({
            "method": method, "fraction": pct,
            "APPA_aleatoric": appa_df["aleatoric_uncertainty"].mean() if appa_df is not None else np.nan,
            "APPA_epistemic": appa_df["epistemic_uncertainty"].mean() if appa_df is not None else np.nan,
            "UTK_aleatoric": utk_df["aleatoric_uncertainty"].mean(),
            "UTK_epistemic": utk_df["epistemic_uncertainty"].mean(),
        })
    df = pd.DataFrame(rows).sort_values(["method", "fraction"])
    df.to_csv(out_csv, index=False)

    methods = sorted(df["method"].unique())
    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 4), sharey=True)
    if len(methods) == 1:
        axes = [axes]
    for ax, method in zip(axes, methods):
        sub = df[df["method"] == method].sort_values("fraction")
        x = np.arange(len(sub))
        width = 0.2
        ax.bar(x - 1.5 * width, sub["APPA_aleatoric"], width, label="APPA aleatoric")
        ax.bar(x - 0.5 * width, sub["UTK_aleatoric"], width, label="UTK aleatoric")
        ax.bar(x + 0.5 * width, sub["APPA_epistemic"], width, label="APPA epistemic")
        ax.bar(x + 1.5 * width, sub["UTK_epistemic"], width, label="UTK epistemic")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{p}%" for p in sub["fraction"]])
        ax.set_title(method)
        ax.set_ylabel("uncertainty (yr²)")
        ax.grid(axis="y", alpha=0.3)
    axes[-1].legend(loc="upper right", fontsize=8)
    fig.suptitle("Aleatoric & Epistemic Uncertainty: APPA-REAL vs UTKFace")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return df


def calibration_per_setting(utk_results, fig_dir):
    rows = []
    for (method, pct), df in sorted(utk_results.items()):
        sigma = df["model_predicted_std"].values
        abs_err = np.abs(df["y_true"].values - df["mean_prediction"].values)
        ece = _expected_calibration_error(sigma, abs_err)
        # Also compute UCE using predictive std (sqrt of predictive variance)
        predictive_std = np.sqrt(df["predictive_uncertainty"].values)
        uce = _expected_calibration_error(predictive_std, abs_err)
        rows.append({"method": method, "fraction": pct, "ECE_modelstd": ece, "UCE_predictive_std": uce})

        binned = _bin_for_calibration(sigma, abs_err)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(binned["sigma_mean"], binned["abs_err_mean"], "o-", label="empirical")
        lo, hi = 0, max(binned["sigma_mean"].max(), binned["abs_err_mean"].max()) * 1.05
        ax.plot([lo, hi], [lo, hi], "k--", label="perfect calibration")
        ax.set_xlabel("predicted σ (years)")
        ax.set_ylabel("empirical |error| (years)")
        ax.set_title(f"UTKFace calibration — {method} @ {pct}%  (ECE={ece:.2f})")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"calibration_{method}_{pct}pct.png"), dpi=150)
        plt.close(fig)
    pd.DataFrame(rows).to_csv(os.path.join(fig_dir, "ece_uce.csv"), index=False)


def calibration_overlay(utk_results, fig_dir, fraction=100):
    fig, ax = plt.subplots(figsize=(6, 6))
    max_v = 0.0
    for (method, pct), df in sorted(utk_results.items()):
        if pct != fraction:
            continue
        sigma = df["model_predicted_std"].values
        abs_err = np.abs(df["y_true"].values - df["mean_prediction"].values)
        binned = _bin_for_calibration(sigma, abs_err)
        ax.plot(binned["sigma_mean"], binned["abs_err_mean"], "o-", label=method)
        max_v = max(max_v, binned["sigma_mean"].max(), binned["abs_err_mean"].max())
    ax.plot([0, max_v * 1.05], [0, max_v * 1.05], "k--", label="perfect")
    ax.set_xlabel("predicted σ (years)")
    ax.set_ylabel("empirical |error| (years)")
    ax.set_title(f"UTKFace calibration — all methods @ {fraction}% data")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"calibration_overlay_{fraction}pct.png"), dpi=150)
    plt.close(fig)


def qualitative_panel(utk_results, utkface_dir, fig_dir, method="ensemble", fraction=100, n=6):
    key = (method, fraction)
    if key not in utk_results:
        print(f"qualitative_panel: no results for {key}; skipping")
        return
    df = utk_results[key].copy()
    df["abs_err"] = np.abs(df["y_true"] - df["mean_prediction"])
    df["sigma"] = df["model_predicted_std"]

    # Pick representative samples by tagged buckets
    picks = []
    # Low σ + low error (confident & correct)
    picks += df.nsmallest(20, "sigma").nsmallest(2, "abs_err").to_dict("records")
    # High σ + high error (correctly uncertain — the desirable failure mode)
    picks += df.nlargest(40, "sigma").nlargest(2, "abs_err").to_dict("records")
    # High error + low σ (overconfident failure — flag this in paper)
    overconfident = df.nlargest(40, "abs_err").nsmallest(2, "sigma").to_dict("records")
    picks += overconfident

    picks = picks[:n]
    if not picks:
        print("qualitative_panel: no samples chosen; skipping")
        return

    cols = 3
    rows = (len(picks) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.8))
    axes = np.atleast_2d(axes).flatten()
    for ax, rec in zip(axes, picks):
        img_path = rec.get("path") or os.path.join(utkface_dir, rec["file_name"])
        try:
            img = Image.open(img_path)
        except Exception as e:
            ax.set_title(f"missing: {e}")
            ax.axis("off")
            continue
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(
            f"true={rec['y_true']:.0f}, pred={rec['mean_prediction']:.1f}\n"
            f"σ={rec['sigma']:.1f}  |err|={rec['abs_err']:.1f}",
            fontsize=9,
        )
    for ax in axes[len(picks):]:
        ax.axis("off")
    fig.suptitle(f"UTKFace qualitative — {method} @ {fraction}%", y=1.02)
    fig.tight_layout()
    out = os.path.join(fig_dir, f"qualitative_panel_{method}_{fraction}pct.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"qualitative panel -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--utkface_dir", required=True)
    ap.add_argument("--utk_results_dir", default=UTKFACE_RESULTS_DIR)
    ap.add_argument("--appa_results_dir", default=APPA_RESULTS_DIR)
    args = ap.parse_args()

    fig_dir = os.path.join(args.utk_results_dir, FIG_DIR_NAME)
    os.makedirs(fig_dir, exist_ok=True)

    appa_results = _read_results(args.appa_results_dir, "_mc_predictions.csv")
    utk_results = _read_results(args.utk_results_dir, "_utkface_predictions.csv")
    if not utk_results:
        raise RuntimeError(f"No UTKFace predictions in {args.utk_results_dir}; run predict_utkface.py first.")

    print("Methods × fractions found:")
    for k in sorted(utk_results):
        print(" ", k)

    metrics_df = build_metrics_table(
        appa_results, utk_results, os.path.join(fig_dir, "cross_dataset_metrics.csv")
    )
    print("\nMetrics table:")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    aleatoric_epistemic_compare(
        appa_results, utk_results,
        os.path.join(fig_dir, "aleatoric_epistemic_compare.csv"),
        os.path.join(fig_dir, "aleatoric_epistemic_compare.png"),
    )
    calibration_per_setting(utk_results, fig_dir)
    calibration_overlay(utk_results, fig_dir, fraction=100)
    qualitative_panel(utk_results, args.utkface_dir, fig_dir, method="ensemble", fraction=100)


if __name__ == "__main__":
    main()
