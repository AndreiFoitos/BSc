import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# =========================
# Configuration
# =========================
RESULTS_DIR = "mc_dropout_results"
OUTPUT_DIR = "uncertainty_visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.4)


# =========================
# Legend handles
# =========================
def uncertainty_legend_handles():
    return [
        Line2D([0], [0], color="C0", lw=4, label="Mean Prediction"),
        Patch(facecolor="C0", alpha=0.25, label="95% CI"),
        Line2D([0], [0], color="k", lw=3, linestyle="--", label="Perfect Truth"),
    ]


# =========================
# Data Loading
# =========================
def load_100percent_models_data(results_dir=RESULTS_DIR):
    all_files = [
        f for f in os.listdir(results_dir)
        if f.endswith("_mc_predictions.csv") and "100percent" in f
    ]

    models_data = {}
    for file in all_files:
        data = pd.read_csv(os.path.join(results_dir, file))

        if "flipout" in file.lower():
            model_type = "Flipout"
        elif "dropconnect" in file.lower():
            model_type = "DropConnect"
        elif "ensemble" in file.lower():
            model_type = "Ensemble"
        else:
            model_type = file.replace("100percent_mc_predictions.csv", "")

        models_data[model_type] = data

    return models_data


# =========================
# Prediction + uncertainty
# =========================
def plot_samples_with_uncertainties_per_model(models_data, output_dir=OUTPUT_DIR):
    from scipy.ndimage import uniform_filter1d

    legend_handles = uncertainty_legend_handles()

    for model_name, data in models_data.items():
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

        y_true = data["y_true"].values
        mean_pred = data["mean_prediction"].values
        ale = np.sqrt(data["aleatoric_uncertainty"].values)
        epi = np.sqrt(data["epistemic_uncertainty"].values)

        idx = np.argsort(y_true)
        y_true, mean_pred, ale, epi = y_true[idx], mean_pred[idx], ale[idx], epi[idx]

        w = max(5, len(y_true) // 100)
        mean_s = uniform_filter1d(mean_pred, w)
        ale_s = uniform_filter1d(ale, w)
        epi_s = uniform_filter1d(epi, w)

        age_min, age_max = y_true.min(), y_true.max()

        for ax, std, unc_name in zip(
            axes,
            [ale_s, epi_s],
            ["Aleatoric Uncertainty", "Epistemic Uncertainty"]
        ):
            ax.plot(y_true, mean_s, linewidth=4)
            ax.fill_between(
                y_true,
                mean_s - 1.96 * std,
                mean_s + 1.96 * std,
                alpha=0.25
            )
            ax.plot([age_min, age_max], [age_min, age_max], "k--", linewidth=3)

            ax.set_title(
                f"{unc_name}",
                fontsize=30,
                fontweight="bold",
                pad=12
            )
            ax.set_xlabel("Apparent Age (Years)", fontsize=28, fontweight="bold")
            ax.set_ylabel("Predicted Age (Years)", fontsize=28, fontweight="bold")

            ax.legend(
                handles=legend_handles,
                fontsize=20,
                frameon=True,
                loc="upper left"
            )

            ax.tick_params(axis="both", labelsize=22)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight("bold")

        out = os.path.join(
            output_dir, f"{model_name.lower()}_uncertainty_predictions.pdf"
        )
        fig.savefig(out, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out}")


# =========================
# Stacked uncertainty components
# =========================
def plot_uncertainty_components_stacked(models_data, output_dir=OUTPUT_DIR):
    legend_handles = uncertainty_legend_handles()

    for model_name, data in models_data.items():
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

        y_true = data["y_true"].values
        mean = data["mean_prediction"].values
        ale = np.sqrt(data["aleatoric_uncertainty"].values)
        epi = np.sqrt(data["epistemic_uncertainty"].values)

        idx = np.argsort(y_true)
        y_true, mean, ale, epi = y_true[idx], mean[idx], ale[idx], epi[idx]

        age_min, age_max = y_true.min(), y_true.max()

        for ax, std, unc_name in zip(
            axes,
            [ale, epi],
            ["Aleatoric Uncertainty", "Epistemic Uncertainty"]
        ):
            ax.plot(y_true, mean, linewidth=3)
            ax.fill_between(y_true, mean - std, mean + std, alpha=0.35)
            ax.fill_between(y_true, mean - 2 * std, mean + 2 * std, alpha=0.2)
            ax.plot([age_min, age_max], [age_min, age_max], "k--", linewidth=3)

            ax.set_title(
                f"{model_name} – {unc_name}",
                fontsize=30,
                fontweight="bold",
                pad=12
            )
            ax.set_xlabel("Apparent Age (Years)", fontsize=28, fontweight="bold")
            ax.set_ylabel("Predicted Age (Years)", fontsize=28, fontweight="bold")

            ax.legend(
                handles=legend_handles,
                fontsize=20,
                frameon=True,
                loc="upper left"
            )

            ax.tick_params(axis="both", labelsize=22)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight("bold")

        out = os.path.join(
            output_dir, f"{model_name.lower()}_uncertainty_components.pdf"
        )
        fig.savefig(out, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out}")


# =========================
# Uncertainty magnitudes
# =========================
def plot_uncertainty_magnitude_per_model(models_data, output_dir=OUTPUT_DIR):
    from scipy.ndimage import uniform_filter1d

    for model_name, data in models_data.items():
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)

        y = data["y_true"].values
        ale = np.sqrt(data["aleatoric_uncertainty"].values)
        epi = np.sqrt(data["epistemic_uncertainty"].values)
        total = np.sqrt(ale**2 + epi**2)
        ratio = epi / (total + 1e-10)

        idx = np.argsort(y)
        y, ale, epi, total, ratio = y[idx], ale[idx], epi[idx], total[idx], ratio[idx]

        w = max(5, len(y) // 100)
        ale = uniform_filter1d(ale, w)
        epi = uniform_filter1d(epi, w)
        total = uniform_filter1d(total, w)
        ratio = uniform_filter1d(ratio, w)

        plots = [
            (ale, "Aleatoric Uncertainty"),
            (epi, "Epistemic Uncertainty"),
            (ratio, "Epistemic / Total"),
            (total, "Total Predictive Uncertainty"),
        ]

        for ax, (vals, unc_name) in zip(axes.flat, plots):
            ax.plot(y, vals, linewidth=4)
            ax.set_title(
                f"{model_name} – {unc_name}",
                fontsize=26,
                fontweight="bold",
                pad=10
            )
            ax.set_xlabel("Apparent Age (Years)", fontsize=26, fontweight="bold")

            ax.tick_params(axis="both", labelsize=20)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight("bold")
            ax.grid(alpha=0.3)

        axes[1, 0].set_ylim(0, 1)

        out = os.path.join(
            output_dir, f"{model_name.lower()}_uncertainty_magnitudes.pdf"
        )
        fig.savefig(out, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out}")


# =========================
# Main
# =========================
if __name__ == "__main__":
    print("Generating per-model uncertainty visualizations...")

    models_data = load_100percent_models_data()
    if not models_data:
        print("No models found.")
        exit(0)

    plot_samples_with_uncertainties_per_model(models_data)
    plot_uncertainty_components_stacked(models_data)
    plot_uncertainty_magnitude_per_model(models_data)

    print(f"\n✓ All figures saved to {OUTPUT_DIR}/")
