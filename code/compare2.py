import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re


file_path = "mc_dropout_results/mc_dropout_inference_summary.csv"
df = pd.read_csv(file_path)

def extract_percentage(name):
    match = re.search(r"(\d+)percent", name)
    return int(match.group(1)) if match else 0

df["Percentage"] = df["model_name"].apply(extract_percentage)

df.sort_values(by="Percentage", ascending=False, inplace=True)


df.set_index("model_name", inplace=True)


plt.figure(figsize=(10, 6))
df['MAE'].plot(kind='bar', colormap='tab20')
plt.title("Model Performance: MAE")
plt.ylabel("MAE Score")
plt.xlabel("Model")
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.tight_layout()
plt.savefig("mae_bar_plot.png")
plt.show()


plt.figure(figsize=(10, 6))
df['RMSE'].plot(kind='bar', colormap='tab20')
plt.title("Model Performance: RMSE")
plt.ylabel("RMSE Score")
plt.xlabel("Model")
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.tight_layout()
plt.savefig("rmse_bar_plot.png")
plt.show()


plt.figure(figsize=(10, 6))
df['MSE'].plot(kind='bar', colormap='tab20')
plt.title("Model Performance: MSE")
plt.ylabel("MSE Score")
plt.xlabel("Model")
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.tight_layout()
plt.savefig("mse_bar_plot.png")
plt.show()


plt.figure(figsize=(10, 6))
df['R2_Score'].plot(kind='bar', colormap='tab20')
plt.title("Model Performance: R² Score")
plt.ylabel("R² Score")
plt.xlabel("Model")
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.tight_layout()
plt.savefig("r2_score_bar_plot.png")
plt.show()


plt.figure(figsize=(10, 6))
df['Mean_Uncertainty'].plot(kind='bar', colormap='tab20')
plt.title("Model Performance: Mean Uncertainty")
plt.ylabel("Mean Uncertainty")
plt.xlabel("Model")
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.tight_layout()
plt.savefig("mean_uncertainty_bar_plot.png")
plt.show()


df_normalized = (df[['MAE', 'RMSE', 'MSE', 'R2_Score', 'Mean_Uncertainty']] - df[['MAE', 'RMSE', 'MSE', 'R2_Score', 'Mean_Uncertainty']].min()) / \
                (df[['MAE', 'RMSE', 'MSE', 'R2_Score', 'Mean_Uncertainty']].max() - df[['MAE', 'RMSE', 'MSE', 'R2_Score', 'Mean_Uncertainty']].min())

plt.figure(figsize=(14, 10))
sns.heatmap(df_normalized, cmap="viridis", linewidths=0.5)
plt.title("Normalized Performance Metrics Heatmap")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("metrics_heatmap_normalized.png")
plt.show()


plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='R2_Score', y='Uncertainty_Error_Correlation', hue=df.index)
plt.title("R² Score vs Uncertainty Error Correlation")
plt.xlabel("R² Score")
plt.ylabel("Uncertainty Error Correlation")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Model")
plt.grid(True)
plt.tight_layout()
plt.savefig("r2_vs_uncertainty_corr.png")
plt.show()
