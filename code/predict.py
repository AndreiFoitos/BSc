import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from model import AgeEstimationModel
from data_loader import AgePredictionDataLoader
import pandas as pd
from sklearn.metrics import r2_score

# --- Config ---
BASE_DIR = "appa-real-release/appa-real-release"
TEST_DIR = os.path.join(BASE_DIR, "test")
TEST_CSV = os.path.join(BASE_DIR, "gt_avg_test.csv")
MODEL_PATH = "age_estimation_model.keras"

# --- Load Data ---
test_loader = AgePredictionDataLoader(TEST_DIR, TEST_CSV, batch_size=32)
test_data = test_loader.get_dataset()

print("Test dataset loaded.")

# --- Load Model ---
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded successfully.")

# --- Evaluation Function ---
def evaluate_with_dataloader(model, dataset):
    actual_ages = []
    predicted_ages = []
    actual_stds = []
    predicted_stds = []
    errors = []

    for batch in dataset:
        images, labels = batch
        preds = model.predict(images, verbose=0)

        pred_avg = preds["apparent_age_avg"].flatten()
        pred_std = preds["apparent_age_std"].flatten()

        actual_avg = labels["apparent_age_avg"].numpy().flatten()
        actual_std = labels["apparent_age_std"].numpy().flatten()

        batch_errors = np.abs(pred_avg - actual_avg)

        errors.extend(batch_errors)
        actual_ages.extend(actual_avg)
        predicted_ages.extend(pred_avg)
        actual_stds.extend(actual_std)
        predicted_stds.extend(pred_std)

    mse = np.mean(np.square(errors))
    r2 = r2_score(actual_ages, predicted_ages)

    print(f"\nEvaluation Results:")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R² Score: {r2:.3f}")

    plot_results(actual_ages, predicted_ages, predicted_stds)
    plot_residuals(actual_ages, predicted_ages)
    plot_residual_histogram(actual_ages, predicted_ages)
    plot_binned_mse(actual_ages, predicted_ages)
    plot_heatmap(actual_ages, predicted_ages)
    plot_calibration_curve(actual_ages, predicted_ages)

# --- Plotting Functions ---
def plot_results(actual_ages, predicted_ages, predicted_stds):
    plt.figure(figsize=(8, 8))
    plt.errorbar(actual_ages, predicted_ages, yerr=predicted_stds, fmt='o', alpha=0.6, label="Predictions with Std Dev")
    plt.plot([min(actual_ages), max(actual_ages)], [min(actual_ages), max(actual_ages)], 'r--', label="Perfect Prediction")
    plt.xlabel("Actual Age")
    plt.ylabel("Predicted Age")
    plt.title("Predicted vs Actual Age with Uncertainty")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_residuals(actual_ages, predicted_ages):
    residuals = np.array(predicted_ages) - np.array(actual_ages)
    plt.figure(figsize=(8, 6))
    plt.scatter(actual_ages, residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Actual Age")
    plt.ylabel("Residual (Predicted - Actual)")
    plt.title("Residual Plot")
    plt.grid(True)
    plt.show()

def plot_residual_histogram(actual_ages, predicted_ages):
    residuals = np.array(predicted_ages) - np.array(actual_ages)
    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=50, color='purple', alpha=0.7)
    plt.axvline(0, color='red', linestyle='--')
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title("Histogram of Residuals")
    plt.grid(True)
    plt.show()

def plot_binned_mse(actual_ages, predicted_ages, bin_size=10):
    actual_ages = np.array(actual_ages)
    predicted_ages = np.array(predicted_ages)
    max_age = int(max(actual_ages)) + bin_size
    bins = range(0, max_age, bin_size)
    bin_centers = []
    mse_per_bin = []

    for start in bins:
        end = start + bin_size
        indices = (actual_ages >= start) & (actual_ages < end)
        if np.sum(indices) > 0:
            errors = predicted_ages[indices] - actual_ages[indices]
            mse_per_bin.append(np.mean(errors ** 2))
            bin_centers.append((start + end) / 2)

    plt.figure(figsize=(8, 6))
    plt.plot(bin_centers, mse_per_bin, marker='o')
    plt.xlabel("Age Bin Center")
    plt.ylabel("MSE")
    plt.title("MSE by Age Bin")
    plt.grid(True)
    plt.show()

def plot_heatmap(actual_ages, predicted_ages):
    plt.figure(figsize=(8, 6))
    plt.hist2d(actual_ages, predicted_ages, bins=50, cmap='plasma')
    plt.colorbar(label='Count')
    plt.plot([0, 100], [0, 100], 'w--')
    plt.xlabel("Actual Age")
    plt.ylabel("Predicted Age")
    plt.title("2D Histogram: Actual vs Predicted")
    plt.grid(True)
    plt.show()

def plot_calibration_curve(actual_ages, predicted_ages, num_bins=10):
    predicted_ages = np.array(predicted_ages)
    actual_ages = np.array(actual_ages)
    bins = np.linspace(min(predicted_ages), max(predicted_ages), num_bins + 1)
    bin_centers = []
    avg_actual = []

    for i in range(num_bins):
        indices = (predicted_ages >= bins[i]) & (predicted_ages < bins[i+1])
        if np.sum(indices) > 0:
            bin_centers.append(np.mean(predicted_ages[indices]))
            avg_actual.append(np.mean(actual_ages[indices]))

    plt.figure(figsize=(8, 6))
    plt.plot(bin_centers, avg_actual, 'o-', label='Model Calibration')
    plt.plot([min(actual_ages), max(actual_ages)], [min(actual_ages), max(actual_ages)], 'r--', label='Perfect Calibration')
    plt.xlabel("Predicted Age")
    plt.ylabel("Average Actual Age")
    plt.title("Calibration Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

# --- Run Evaluation ---
evaluate_with_dataloader(model, test_data)

# --- Export Predictions ---
def export_predictions_to_csv(model, dataset, original_csv_path, output_csv_path="predictions_with_outputs.csv"):
    df = pd.read_csv(original_csv_path)

    predicted_ages = []
    predicted_stds = []

    file_names = df['file_name'].tolist()
    batch_index = 0

    for batch in dataset:
        images, labels = batch
        preds = model.predict(images, verbose=0)

        pred_avg = preds["apparent_age_avg"].flatten()
        pred_std = preds["apparent_age_std"].flatten()

        predicted_ages.extend(pred_avg.tolist())
        predicted_stds.extend(pred_std.tolist())

        batch_index += 1

    if len(predicted_ages) != len(df):
        print("Warning: Mismatch between number of predictions and CSV entries. Truncating to shortest.")
        min_len = min(len(predicted_ages), len(df))
        df = df.iloc[:min_len]
        predicted_ages = predicted_ages[:min_len]
        predicted_stds = predicted_stds[:min_len]

    df["predicted_age"] = predicted_ages
    df["predicted_std"] = predicted_stds

    df.to_csv(output_csv_path, index=False)
    print(f"\nPredictions saved to {output_csv_path}")

# --- Save Predictions to CSV ---
export_predictions_to_csv(model, test_data, TEST_CSV)
