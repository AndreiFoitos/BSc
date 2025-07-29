import os
import re
import pandas as pd
import matplotlib.pyplot as plt

INPUT_DIR = "mc_dropout_results"
OUTPUT_DIR = "compare"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_model_info(filename):
    match = re.match(r'(flipout|dropconnect|ensemble)(\d+)percent_mc_predictions\.csv', filename)
    if match:
        model_type, percentage = match.groups()
        return model_type, int(percentage)
    
    match = re.match(r'(flipout)(\d+)percentlastlayer_mc_predictions\.csv', filename)
    if match:
        model_type, percentage = match.groups()
        return model_type + "_lastlayer", int(percentage)

    return None, None

data_by_model = {
    'flipout': [],
    'dropconnect': [],
    'ensemble': []
}

for file in os.listdir(INPUT_DIR):
    if file.endswith("_mc_predictions.csv"):
        model_type, percent = extract_model_info(file)
        if model_type:
            print(f"Processing file: {file}, model: {model_type}, percent: {percent}")
            df = pd.read_csv(os.path.join(INPUT_DIR, file))
            avg_aleatoric = df["aleatoric_uncertainty"].mean()
            avg_epistemic = df["epistemic_uncertainty"].mean()
            avg_predictive = df["predictive_uncertainty"].mean()
            avg_model_std = df["model_predicted_std"].mean()
            if model_type in data_by_model:
                data_by_model[model_type].append({
                    "percentage": percent,
                    "aleatoric": avg_aleatoric,
                    "epistemic": avg_epistemic,
                    "predictive": avg_predictive,
                    "model_std": avg_model_std
                })

def plot_uncertainty_vs_true_age(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR, bin_width=5):
    result_files = [f for f in os.listdir(input_dir) if f.endswith("_mc_predictions.csv")]

    for file in result_files:
        df = pd.read_csv(os.path.join(input_dir, file))
        model_name = file.replace("_mc_predictions.csv", "")

        max_age = df["y_true"].max()
        bins = list(range(0, int(max_age) + bin_width, bin_width))
        df["age_bin"] = pd.cut(df["y_true"], bins=bins, right=False)

        grouped = df.groupby("age_bin").agg({
            "aleatoric_uncertainty": "mean",
            "epistemic_uncertainty": "mean"
        }).reset_index()

        bin_midpoints = grouped["age_bin"].apply(lambda x: x.left + bin_width / 2)

        plt.figure(figsize=(10, 7))
        plt.plot(bin_midpoints, grouped["aleatoric_uncertainty"], label="Aleatoric", marker='o', markersize=8, linewidth=2)
        plt.plot(bin_midpoints, grouped["epistemic_uncertainty"], label="Epistemic", marker='s', markersize=8, linewidth=2, linestyle='--')

        plt.xlabel("Apparent Age", fontsize=14)
        plt.ylabel("Average Uncertainty (Years)", fontsize=14)
        plt.title(f"Uncertainty vs Age — {model_name}", fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"{model_name}_uncertainty_vs_age.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved: {save_path}")

fixed_ticks = [1, 5, 10, 25, 50, 75, 100]

plot_uncertainty_vs_true_age()


ranking_df = pd.read_csv(r"C:\Users\Andrei\Documents\GitHub\BSc\code\mc_dropout_results\model_ranking.csv")

def extract_model_info_from_name(name):
    match = re.match(r'(flipout|dropconnect|ensemble)(\d+)percent', name)
    if match:
        model_type, percent = match.groups()
        return model_type, int(percent)
    return None, None

ranking_df[['model_type', 'percentage']] = ranking_df['model_name'].apply(
    lambda x: pd.Series(extract_model_info_from_name(x))
)

for model_type, data in data_by_model.items():
    if not data:
        print(f"No data found for model type: {model_type}")
        continue

    df_uncertainty = pd.DataFrame(data).sort_values("percentage")
    df_metrics = ranking_df[ranking_df['model_type'] == model_type][['percentage', 'MAE', 'RMSE']]

    merged_df = pd.merge(df_uncertainty, df_metrics, on='percentage', how='left')

    plt.figure(figsize=(10, 6))
    plt.plot(merged_df["percentage"], merged_df["aleatoric"], marker='o', linestyle='-', label="Aleatoric",
             markersize=8, linewidth=2)
    plt.plot(merged_df["percentage"], merged_df["epistemic"], marker='s', linestyle='--', label="Epistemic",
             markersize=8, linewidth=2)
    plt.plot(merged_df["percentage"], merged_df["MAE"], marker='^', linestyle='-.', label="MAE",
             markersize=8, linewidth=2, color='red')
    plt.plot(merged_df["percentage"], merged_df["RMSE"], marker='d', linestyle=':', label="RMSE",
             markersize=8, linewidth=2, color='purple')

    title_name = model_type.replace("_", " ").capitalize()
    plt.title(f"{title_name} — Uncertainty, MAE & RMSE vs Training Data Percentage", fontsize=16, fontweight='bold')
    plt.xlabel("Training Data Percentage (%)", fontsize=14)
    plt.ylabel("Value (Years)", fontsize=14)
    plt.xticks(fixed_ticks, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, f"{model_type}_uncertainty_mae_rmse_vs_percentage.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved plot with MAE and RMSE: {save_path}")
