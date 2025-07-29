import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Configuration ===
INPUT_DIR = "mc_dropout_results"
output_folder = r"C:\Users\Andrei\Documents\GitHub\BSc\plots"
os.makedirs(output_folder, exist_ok=True)

# Seaborn style
sns.set(style="whitegrid")

# === Filename parsing ===
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

# === Load data ===
data_by_model = {}

for file in os.listdir(INPUT_DIR):
    if file.endswith("_mc_predictions.csv"):
        model_type, percent = extract_model_info(file)
        if model_type:
            print(f"Processing file: {file}, model: {model_type}, percent: {percent}")
            df = pd.read_csv(os.path.join(INPUT_DIR, file))

            avg_aleatoric = df["aleatoric_uncertainty"].mean()
            avg_epistemic = df["epistemic_uncertainty"].mean()

            if model_type not in data_by_model:
                data_by_model[model_type] = []
            data_by_model[model_type].append({
                "percentage": percent,
                "aleatoric": avg_aleatoric,
                "epistemic": avg_epistemic
            })

# === Plotting ===
fixed_ticks = [1, 5, 10, 25, 50, 75, 100]

for model_type, data in data_by_model.items():
    df = pd.DataFrame(data).sort_values("percentage")

    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Aleatoric: Primary y-axis
    ax1.plot(df["percentage"], df["aleatoric"], marker='o', linestyle='-', color='mediumseagreen',
             label="Aleatoric", markersize=10, linewidth=3)
    ax1.set_xlabel("Training Data Percentage (%)", fontsize=30, fontweight='bold', color='black')
    ax1.set_ylabel("Aleatoric (Years²)", fontsize=30, fontweight='bold', color='black')
    ax1.set_xticks(fixed_ticks)
    ax1.tick_params(axis='y', labelcolor='black', labelsize=25)
    ax1.tick_params(axis='x', labelcolor='black', labelsize=25)
    for tick in ax1.get_xticklabels() + ax1.get_yticklabels():
        tick.set_fontweight('bold')
        tick.set_color('black')
    ax1.grid(True)

    # Epistemic: Secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(df["percentage"], df["epistemic"], marker='s', linestyle='--', color='steelblue',
             label="Epistemic", markersize=10, linewidth=3)
    ax2.set_ylabel("Epistemic (Years²)", fontsize=30, fontweight='bold', color='black')
    ax2.tick_params(axis='y', labelcolor='black', labelsize=25)
    for tick in ax2.get_yticklabels():
        tick.set_fontweight('bold')
        tick.set_color('black')

    # Legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="center right", fontsize=30, title_fontsize=13, frameon=True)

    # Layout
    plt.tight_layout()

    # Save
    save_path = os.path.join(output_folder, f"{model_type}_epistemic_aleatoric_vs_percentage.pdf")
    plt.savefig(save_path, format='pdf')
    print(f"Saved: {save_path}")
    plt.close()


    
