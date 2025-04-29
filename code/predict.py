import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
import random

from model import AgeEstimationModel

# --- Seeding for Reproducibility ---
SEED = 36
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# --- Paths (Adjust if needed) ---
MODELS_DIR = "trained_models_by_fraction"
SAVE_RESULTS_DIR = "mc_dropout_results"

os.makedirs(SAVE_RESULTS_DIR, exist_ok=True)

# --- Constants ---
BATCH_SIZE = 32
MC_SAMPLES = 20

TEST_DIR = "C:/Users/Andrei/Documents/GitHub/BSc/appa-real-release/appa-real-release/test"
TEST_CSV = "C:/Users/Andrei/Documents/GitHub/BSc/appa-real-release/appa-real-release/gt_avg_test.csv"

# --- Load Test Data ---
def load_test_data(test_dir, test_csv_path, img_size=(224, 224), batch_size=32):
    df = pd.read_csv(test_csv_path)
    df["face_file_name"] = df["file_name"].apply(lambda x: f"{x}_face.jpg")
    df["face_path"] = df["face_file_name"].apply(lambda fname: os.path.join(test_dir, fname))
    df = df[df["face_path"].apply(os.path.exists)].reset_index(drop=True)

    image_paths = df["face_path"].tolist()
    apparent_age_avg = df["real_age"].astype(np.float32).values
    apparent_age_std = df["apparent_age_std"].astype(np.float32).values

    def load_and_preprocess_image(path):
        img = tf.keras.utils.load_img(path, target_size=img_size)
        img_array = tf.keras.utils.img_to_array(img)
        img_array = img_array / 255.0
        return img_array

    images = np.array([load_and_preprocess_image(path) for path in image_paths])
    labels = {
        "apparent_age_avg": apparent_age_avg,
        "apparent_age_std": apparent_age_std
    }

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.batch(batch_size, drop_remainder=False)
    return dataset

# --- MC Dropout Inference ---
# --- MC Dropout Inference with Progress Tracking --- 
def mc_inference(models, dataset, n_samples=20):
    all_means = []
    all_vars = []
    all_model_stds = []
    y_trues = []

    # Progress bar for MC sampling
    with tqdm(total=n_samples, desc="MC Sampling Passes") as mc_pbar:
        for _ in range(n_samples):
            sample_means = []
            sample_vars = []
            sample_stds = []
            sample_trues = []

            # Progress bar for dataset batches
            with tqdm(total=61, desc="Batch Processing", leave=False) as batch_pbar:
                for images, labels in dataset:
                    batch_means = []
                    batch_vars = []
                    batch_stds = []

                    # Progress bar for models in each batch
                    for model in models:
                        preds = model(images, training=True)
                        mean = preds["apparent_age_avg"].numpy().flatten()
                        std = preds["apparent_age_std"].numpy().flatten()
                        var = np.square(std)

                        batch_means.append(mean)
                        batch_vars.append(var)
                        batch_stds.append(std)

                    batch_mean_avg = np.mean(batch_means, axis=0)
                    batch_var_avg = np.mean(batch_vars, axis=0)
                    batch_std_avg = np.mean(batch_stds, axis=0)

                    sample_means.append(batch_mean_avg)
                    sample_vars.append(batch_var_avg)
                    sample_stds.append(batch_std_avg)
                    sample_trues.append(labels["apparent_age_avg"].numpy().flatten())

                    batch_pbar.update(1)  # Update batch progress bar

            all_means.append(np.concatenate(sample_means))
            all_vars.append(np.concatenate(sample_vars))
            all_model_stds.append(np.concatenate(sample_stds))

            if len(y_trues) == 0:
                y_trues = np.concatenate(sample_trues)

            mc_pbar.update(1)  # Update MC sampling progress bar

    # Stack the results from each MC sampling pass
    all_means = np.stack(all_means, axis=0)
    all_vars = np.stack(all_vars, axis=0)
    all_model_stds = np.stack(all_model_stds, axis=0)

    # Compute final predictions and uncertainties
    pred_mean = np.mean(all_means, axis=0)
    aleatoric = np.mean(all_vars, axis=0)
    epistemic = np.var(all_means, axis=0)
    predictive = aleatoric + epistemic
    pred_model_std = np.mean(all_model_stds, axis=0)

    return pred_mean, aleatoric, epistemic, predictive, pred_model_std, y_trues


# --- Plotting Functions ---
def plot_predictions(y_true, mean, std, save_path_prefix):
    os.makedirs(os.path.dirname(save_path_prefix), exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.errorbar(y_true, mean, yerr=std, fmt='o', ecolor='lightgray', alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
    plt.xlabel("True Age")
    plt.ylabel("Predicted Age")
    plt.title("MC Dropout Predictions with Uncertainty")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path_prefix + "_scatter.png")
    plt.close()

    abs_errors = np.abs(y_true - mean)
    plt.figure(figsize=(8, 6))
    plt.scatter(std, abs_errors, alpha=0.5)
    plt.xlabel("Uncertainty (std)")
    plt.ylabel("Absolute Error")
    plt.title("Uncertainty vs Error")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path_prefix + "_uncertainty_vs_error.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.hist(std, bins=30, color="skyblue", edgecolor="black")
    plt.xlabel("Uncertainty (std)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Prediction Uncertainty")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path_prefix + "_uncertainty_histogram.png")
    plt.close()

# --- Load Test Dataset Once ---
print("Loading test dataset...")
test_dataset = load_test_data(TEST_DIR, TEST_CSV, batch_size=BATCH_SIZE)


# --- Main Inference Loop ---

# Optional color codes for nicer terminal output (can remove if not wanted)
BLUE = '\033[94m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
ENDC = '\033[0m'

model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".keras")]

for model_file in model_files:

    model_path = os.path.join(MODELS_DIR, model_file)
    model_name = model_file.replace(".keras", "")

    with tqdm(total=4, desc=f"🚀 {model_name}", leave=True) as pbar:
        test_dataset = load_test_data(TEST_DIR, TEST_CSV, batch_size=BATCH_SIZE)

        print(f"\n{BLUE}🔵 [START] Processing model: {model_name}{ENDC}")

        # Step 1: Load model
        print(f"📦 Loading model: {model_name}")
        model = tf.keras.models.load_model(model_path, compile=False)
        pbar.update(1)

        # Step 2: Run MC Dropout inference
        print(f"🛠️  Running MC Dropout inference for {model_name}...")
        pred_mean, aleatoric, epistemic, predictive, pred_model_std, y_true = mc_inference(
            [model],
            test_dataset,
            n_samples=MC_SAMPLES
        )
        pbar.update(1)

        # Step 3: Save predictions
        save_path = os.path.join(SAVE_RESULTS_DIR, f"{model_name}_mc_predictions.csv")
        df = pd.DataFrame({
            "y_true": y_true,
            "mean_prediction": pred_mean,
            "aleatoric_uncertainty": aleatoric,
            "epistemic_uncertainty": epistemic,
            "predictive_uncertainty": predictive,
            "model_predicted_std": pred_model_std
        })
        df.to_csv(save_path, index=False)
        print(f"Predictions saved: {save_path}")
        pbar.update(1)

        # Step 4: Generate plots
        plot_prefix = os.path.join(SAVE_RESULTS_DIR, model_name)
        print(f"Generating plots for {model_name}...")
        plot_predictions(y_true, pred_mean, pred_model_std, plot_prefix)
        pbar.update(1)

        print(f"{GREEN} [DONE] Finished processing {model_name}{ENDC}\n" + "-"*60)


# --- Generate Inference Report ---
def generate_inference_report(results_dir=SAVE_RESULTS_DIR, output_csv="mc_dropout_inference_summary.csv"):
    rows = []
    result_files = [f for f in os.listdir(results_dir) if f.endswith("_mc_predictions.csv")]

    for file in result_files:
        data = pd.read_csv(os.path.join(results_dir, file))

        y_true = data['y_true'].values
        mean_pred = data['mean_prediction'].values
        std_pred = data['model_predicted_std'].values

        mae = mean_absolute_error(y_true, mean_pred)
        mse = mean_squared_error(y_true, mean_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, mean_pred)
        mean_uncertainty = np.mean(std_pred)

        abs_errors = np.abs(y_true - mean_pred)
        corr = pearsonr(abs_errors, std_pred)[0] if len(abs_errors) > 1 else np.nan

        rows.append({
            "model_name": file.replace("_mc_predictions.csv", ""),
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2_Score": r2,
            "Mean_Uncertainty": mean_uncertainty,
            "Uncertainty_Error_Correlation": corr
        })

    df = pd.DataFrame(rows)
    df.sort_values("model_name", inplace=True)
    output_path = os.path.join(results_dir, output_csv)
    df.to_csv(output_path, index=False)
    print(f"\nInference report saved to {output_path}")

    plot_path = os.path.join(results_dir, "mae_vs_uncertainty.png")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="Mean_Uncertainty", y="MAE", hue="model_name", palette="tab10", s=100)
    plt.title("MAE vs Mean Uncertainty Across Models")
    plt.xlabel("Mean Uncertainty (std)")
    plt.ylabel("Mean Absolute Error (MAE)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"MAE vs Uncertainty plot saved to {plot_path}")

# --- Plot Calibration Curves ---
def plot_calibration_curves(results_dir=SAVE_RESULTS_DIR):
    result_files = [f for f in os.listdir(results_dir) if f.endswith("_mc_predictions.csv")]
    plt.figure(figsize=(10, 8))

    for file in result_files:
        data = pd.read_csv(os.path.join(results_dir, file))

        y_true = data['y_true'].values
        mean_pred = data['mean_prediction'].values
        std_pred = data['model_predicted_std'].values

        abs_error = np.abs(mean_pred - y_true)

        bins = np.percentile(std_pred, np.linspace(0, 100, 10))
        bin_indices = np.digitize(std_pred, bins)

        bin_uncertainty_means = []
        bin_error_means = []

        for b in range(1, len(bins)):
            if np.any(bin_indices == b):
                bin_uncertainty_means.append(std_pred[bin_indices == b].mean())
                bin_error_means.append(abs_error[bin_indices == b].mean())

        bin_uncertainty_means = np.array(bin_uncertainty_means)
        bin_error_means = np.array(bin_error_means)
        valid = ~np.isnan(bin_uncertainty_means) & ~np.isnan(bin_error_means)

        model_name = file.replace("_mc_predictions.csv", "")
        plt.plot(bin_uncertainty_means[valid], bin_error_means[valid], marker='o', label=model_name)

    plt.plot([0, plt.xlim()[1]], [0, plt.xlim()[1]], 'k--', label='Perfect Calibration')
    plt.xlabel('Predicted Uncertainty (std)')
    plt.ylabel('Empirical Error (MAE)')
    plt.title('Calibration Curves')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "calibration_curves.png"))
    plt.close()
    print(f"Calibration curves saved to {os.path.join(results_dir, 'calibration_curves.png')}")

# --- Model Ranking ---
def rank_models(results_dir=SAVE_RESULTS_DIR, input_csv="mc_dropout_inference_summary.csv", output_csv="model_ranking.csv", top_n=5):
    df = pd.read_csv(os.path.join(results_dir, input_csv))

    weights = {
        'MAE': 0.4,
        'RMSE': 0.3,
        'R2_Score': -0.2,
        'Mean_Uncertainty': 0.1,
    }

    for col in weights:
        if col in df.columns:
            min_val, max_val = df[col].min(), df[col].max()
            df[f"{col}_norm"] = (df[col] - min_val) / (max_val - min_val) if (max_val - min_val) > 1e-6 else 0.0

    df['composite_score'] = sum(weights[col] * df[f"{col}_norm"] for col in weights)

    df.sort_values('composite_score', inplace=True)
    df.to_csv(os.path.join(results_dir, output_csv), index=False)

    print(f"\n Model ranking saved to {os.path.join(results_dir, output_csv)}")
    print("\n🏆 Top Models:")
    print(df[['model_name', 'composite_score']].head(top_n))

    return df

def plot_model_rankings(ranked_df, results_dir=SAVE_RESULTS_DIR, plot_filename="model_ranking_plot.png"):
    sns.set(style="whitegrid")
    ranked_df = ranked_df.sort_values('composite_score')

    plt.figure(figsize=(12, 6))
    sns.barplot(x='composite_score', y='model_name', data=ranked_df, palette='viridis')

    plt.title('Model Ranking Based on Composite Score', fontsize=16)
    plt.xlabel('Composite Score (Lower is Better)', fontsize=14)
    plt.ylabel('Model Name', fontsize=14)

    plt.tight_layout()
    save_path = os.path.join(results_dir, plot_filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Model ranking plot saved to {save_path}")

# --- RUN ---
generate_inference_report()
plot_calibration_curves()
ranked_models = rank_models()
plot_model_rankings(ranked_models)
