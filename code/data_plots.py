import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from model import AgeEstimationModel
from data_loader import AgePredictionDataLoader


TRAIN_CSV = r"C:\Users\Andrei\Documents\GitHub\BSc\appa-real-release\appa-real-release\gt_avg_train.csv"



import pandas as pd

def evaluate_apparent_vs_real(csv_path):
    df = pd.read_csv(csv_path)

    actual_ages = df["real_age"].values
    apparent_ages = df["apparent_age_avg"].values
    apparent_stds = df["apparent_age_std"].values

    errors = np.abs(apparent_ages - actual_ages)
    mae = np.mean(errors)
    mse = np.mean(np.square(errors))

    print(f"\nEvaluation of Apparent vs Real Age:")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")

    plot_results(actual_ages, apparent_ages, apparent_stds)
    plot_residuals(actual_ages, apparent_ages)
    plot_residual_histogram(actual_ages, apparent_ages)
    plot_binned_mae(actual_ages, apparent_ages)
    plot_heatmap(actual_ages, apparent_ages)
    plot_calibration_curve(actual_ages, apparent_ages)


# --- Plotting Functions ---
def plot_results(actual_ages, predicted_ages, predicted_stds):
    plt.figure(figsize=(8, 8))
    plt.errorbar(actual_ages, predicted_ages, yerr=predicted_stds, fmt='o', alpha=0.6, label="Predictions with Std Dev")
    plt.plot([min(actual_ages), max(actual_ages)], [min(actual_ages), max(actual_ages)], 'r--', label="Perfect Prediction")
    plt.xlabel("Average Apparent Age")
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

def plot_binned_mae(actual_ages, predicted_ages, bin_size=10):
    actual_ages = np.array(actual_ages)
    predicted_ages = np.array(predicted_ages)
    max_age = int(max(actual_ages)) + bin_size
    bins = range(0, max_age, bin_size)
    bin_centers = []
    mae_per_bin = []

    for start in bins:
        end = start + bin_size
        indices = (actual_ages >= start) & (actual_ages < end)
        if np.sum(indices) > 0:
            errors = np.abs(predicted_ages[indices] - actual_ages[indices])
            mae_per_bin.append(np.mean(errors))
            bin_centers.append((start + end) / 2)

    plt.figure(figsize=(8, 6))
    plt.plot(bin_centers, mae_per_bin, marker='o')
    plt.xlabel("Age Bin Center")
    plt.ylabel("MAE")
    plt.title("MAE by Age Bin")
    plt.grid(True)
    plt.show()

def plot_heatmap(actual_ages, predicted_ages):
    plt.figure(figsize=(8, 6))
    plt.hist2d(actual_ages, predicted_ages, bins=50, cmap='plasma')
    plt.colorbar(label='Count')
    plt.plot([0, 100], [0, 100], 'w--')
    plt.xlabel("Average Apparent Age")
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
    plt.xlabel("Real Age")
    plt.ylabel("Average Apparent Age")
    plt.title("Calibration Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

evaluate_apparent_vs_real(TRAIN_CSV)
