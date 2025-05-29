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

        plt.figure(figsize=(10, 6))
        plt.plot(bin_midpoints, grouped["aleatoric_uncertainty"], label="Aleatoric", marker='o')
        plt.plot(bin_midpoints, grouped["epistemic_uncertainty"], label="Epistemic", marker='s')

        plt.xlabel("Apparent Age")
        plt.ylabel("Average Uncertainty (Years)")
        plt.title(f"Uncertainty vs Age (Grouped) — {model_name}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        save_path = os.path.join(output_dir, f"{model_name}_uncertainty_vs_age.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")

fixed_ticks = [1, 5, 10, 25, 50, 75, 100]

plot_uncertainty_vs_true_age()

for model_type, data in data_by_model.items():
    if not data:
        print(f"No data found for model type: {model_type}")
        continue

    df = pd.DataFrame(data).sort_values("percentage")

    plt.figure(figsize=(8, 5))
    plt.plot(df["percentage"], df["aleatoric"], marker='o', linestyle='-', label="Aleatoric")
    plt.plot(df["percentage"], df["epistemic"], marker='s', linestyle='--', label="Epistemic")

    title_name = model_type.replace("_", " ").capitalize()
    plt.title(f"{title_name} - Uncertainty vs Data Percentage")
    plt.xlabel("Training Data Percentage (%)")
    plt.ylabel("Average Uncertainty (Years)")
    plt.xticks(fixed_ticks)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, f"{model_type}_uncertainty_vs_percentage.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot: {save_path}")
