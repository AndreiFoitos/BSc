import pandas as pd
import numpy as np

def find_suspicious_predictions(csv_path, large_error_threshold=20, low_prediction_threshold=5):
    df = pd.read_csv(csv_path)

    if "predicted_age" not in df.columns or "apparent_age_avg" not in df.columns:
        print("Required columns not found in CSV.")
        return

    df["error"] = np.abs(df["predicted_age"] - df["apparent_age_avg"])

    large_errors = df[df["error"] >= large_error_threshold]
    low_predictions = df[df["predicted_age"] < low_prediction_threshold]

    print(f"\n--- Suspicious Predictions Report ---")
    print(f"\nPredictions with error >= {large_error_threshold} years:")
    print(large_errors[["file_name", "apparent_age_avg", "predicted_age", "error"]])

    print(f"\nPredictions with predicted age < {low_prediction_threshold}:")
    print(low_predictions[["file_name", "apparent_age_avg", "predicted_age", "error"]])

# --- Run Suspicious Prediction Search ---
find_suspicious_predictions("predictions_with_outputs.csv")
