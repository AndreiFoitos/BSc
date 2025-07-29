import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Load CSV data ===
df = pd.read_csv(r"C:\Users\Andrei\Documents\GitHub\BSc\code\mc_dropout_results\model_ranking.csv")  # Update with your actual CSV path

# === Parse percentage and model type ===
df['percentage'] = df['model_name'].str.extract(r'(\d+)').astype(int)
df['model_type'] = df['model_name'].str.extract(r'([a-zA-Z]+)')
df = df.sort_values(by='percentage')

# === Output folder setup ===
output_folder = r"C:\Users\Andrei\Documents\GitHub\BSc\plots"
os.makedirs(output_folder, exist_ok=True)

# === Plotting config ===
sns.set(style="whitegrid")
metric_labels = {
    'MAE': 'Mean Absolute Error (MAE)',
    'RMSE': 'Root Mean Square Error (RMSE)',
    'R2_Score': 'R² Score'
}

for metric, ylabel in metric_labels.items():
    plt.figure(figsize=(14, 8))
    
    for model in df['model_type'].unique():
        subset = df[df['model_type'] == model]
        sns.lineplot(x='percentage', y=metric, data=subset, marker='o', label=model, linewidth=3)
    
    # Axis labels and formatting
    plt.xlabel("Training Data Percentage (%)", fontsize=30, fontweight='bold')
    plt.ylabel(ylabel, fontsize=30, fontweight='bold')
    plt.xticks(fontsize=25, fontweight='bold')
    plt.yticks(fontsize=25, fontweight='bold')
    plt.legend(fontsize=30, title_fontsize=13, frameon=True)

    # Save the figure
    filename = f"{metric.lower()}_vs_percentage_by_model.pdf"
    output_path = os.path.join(output_folder, filename)
    plt.tight_layout()
    plt.savefig(output_path, format='pdf')
    plt.show()