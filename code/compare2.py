import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV
df = pd.read_csv("mc_dropout_results\model_ranking.csv")

# Select the metrics you want to plot
metrics = ["MAE", "RMSE", "R2_Score", "Mean_Uncertainty", "Uncertainty_Error_Correlation"]

# Melt the DataFrame so we can plot with model_name on x and metric values as bars
df_melted = df.melt(id_vars=["model_name"], value_vars=metrics,
                    var_name="Metric", value_name="Value")

# Set plot size
plt.figure(figsize=(14, 6))

# Create grouped bar chart
import seaborn as sns
sns.set(style="whitegrid")

sns.barplot(data=df_melted, x="model_name", y="Value", hue="Metric", palette="Set2")

# Rotate x-axis labels for readability
plt.xticks(rotation=45, ha='right')

plt.xlabel("Model Name")
plt.ylabel("Metric Value")
plt.title("Model Comparison Across Metrics")
plt.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
