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
    return None, None

# Include ensemble
data_by_model = {'flipout': [], 'dropconnect': [], 'ensemble': []}

for file in os.listdir(INPUT_DIR):
    if file.endswith("_mc_predictions.csv"):
        model_type, percent = extract_model_info(file)
        if model_type:
            print(f"Processing file: {file} → model: {model_type}, percent: {percent}")
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

def plot_uncertainty_vs_true_age(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR):
    result_files = [f for f in os.listdir(input_dir) if f.endswith("_mc_predictions.csv")]

    for file in result_files:
        df = pd.read_csv(os.path.join(input_dir, file))
        model_name = file.replace("_mc_predictions.csv", "")

        y_true = df['y_true']
        aleatoric = df['aleatoric_uncertainty']
        epistemic = df['epistemic_uncertainty']
        predictive = df['predictive_uncertainty']
        model_std = df['model_predicted_std']

        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, aleatoric, alpha=0.4, label="Aleatoric", s=10)
        plt.scatter(y_true, epistemic, alpha=0.4, label="Epistemic", s=10)
        plt.scatter(y_true, predictive, alpha=0.4, label="Predictive", s=10)
        plt.scatter(y_true, model_std, alpha=0.4, label="Model Std", s=10)

        plt.xlabel("True Age")
        plt.ylabel("Uncertainty")
        plt.title(f"Uncertainty vs True Age — {model_name}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        save_path = os.path.join(output_dir, f"{model_name}_uncertainty_vs_true_age.png")
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

    for metric in ["aleatoric", "epistemic", "predictive", "model_std"]:
        plt.figure(figsize=(8, 5))
        plt.plot(df["percentage"], df[metric], marker='o', linestyle='-')

        plt.title(f"{model_type.capitalize()} - {metric.replace('_', ' ').title()} vs Data Percentage")
        plt.xlabel("Training Data Percentage (%)")
        plt.ylabel(f"Average {metric.replace('_', ' ').title()}")
        plt.xticks(fixed_ticks)
        plt.grid(True)
        plt.tight_layout()

        save_path = os.path.join(OUTPUT_DIR, f"{model_type}_{metric}_vs_percentage.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved plot: {save_path}")
