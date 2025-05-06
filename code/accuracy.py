import os
import pandas as pd
import numpy as np

# Directory with saved *_mc_predictions.csv files
RESULTS_DIR = "mc_dropout_results"  # Adjust if needed

summary = []

for file in os.listdir(RESULTS_DIR):
    if file.endswith("_mc_predictions.csv"):
        path = os.path.join(RESULTS_DIR, file)
        df = pd.read_csv(path)

        # Compute relative accuracy: (1 - |pred - true| / true) * 100
        relative_accuracy = (1 - np.abs(df["mean_prediction"] - df["y_true"]) / df["y_true"]) * 100
        relative_accuracy = np.clip(relative_accuracy, 0, 100)

        summary.append({
            "model_name": file.replace("_mc_predictions.csv", ""),
            "Mean_Relative_Accuracy (%)": relative_accuracy.mean()
        })

# Create summary DataFrame
summary_df = pd.DataFrame(summary)
summary_df.sort_values("Mean_Relative_Accuracy (%)", ascending=False, inplace=True)

# Save to new CSV
output_csv = os.path.join(RESULTS_DIR, "relative_accuracy_summary.csv")
summary_df.to_csv(output_csv, index=False)

print(f"Summary saved to {output_csv}")
