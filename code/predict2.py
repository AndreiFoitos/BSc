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
import re
from model import AgeEstimationModel

SEED = 36
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

tf.config.run_functions_eagerly(True)
print("Eager execution:", tf.executing_eagerly())


MODELS_DIR = "trained_models_by_fraction"
SAVE_RESULTS_DIR = "mc_dropout_results"
TEST_DIR = "C:/Users/Andrei/Documents/GitHub/BSc/appa-real-release/appa-real-release/test"
TEST_CSV = "C:/Users/Andrei/Documents/GitHub/BSc/appa-real-release/appa-real-release/gt_avg_test.csv"
BATCH_SIZE = 32
MC_SAMPLES = 20

os.makedirs(SAVE_RESULTS_DIR, exist_ok=True)

def load_test_data(test_dir, test_csv_path, img_size=(224, 224), batch_size=32):
    df = pd.read_csv(test_csv_path)
    df["face_file_name"] = df["file_name"].apply(lambda x: f"{x}_face.jpg")
    df["face_path"] = df["face_file_name"].apply(lambda fname: os.path.join(test_dir, fname))
    df = df[df["face_path"].apply(os.path.exists)].reset_index(drop=True)

    image_paths = df["face_path"].tolist()
    apparent_age_avg = df["apparent_age_avg"].astype(np.float32).values
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

def mc_inference(models, dataset, n_samples=20):
    all_means = []
    all_vars = []
    all_model_stds = []
    y_trues = []

    with tqdm(total=n_samples, desc="MC Sampling Passes", dynamic_ncols=True) as mc_pbar:
        for i in range(n_samples):
            print(f"Running MC sample {i+1}/{n_samples}")
            sample_means = []
            sample_vars = []
            sample_stds = []
            sample_trues = []

            for images, labels in dataset:
                batch_means = []
                batch_vars = []
                batch_stds = []

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

            all_means.append(np.concatenate(sample_means))
            all_vars.append(np.concatenate(sample_vars))
            all_model_stds.append(np.concatenate(sample_stds))

            if len(y_trues) == 0:
                y_trues = np.concatenate(sample_trues)

            mc_pbar.update(1)
            mc_pbar.refresh()

    all_means = np.stack(all_means, axis=0)
    all_vars = np.stack(all_vars, axis=0)
    all_model_stds = np.stack(all_model_stds, axis=0)

    pred_mean = np.mean(all_means, axis=0)
    aleatoric = np.mean(all_vars, axis=0)
    epistemic = np.var(all_means, axis=0)
    predictive = aleatoric + epistemic
    pred_model_std = np.mean(all_model_stds, axis=0)

    return pred_mean, aleatoric, epistemic, predictive, pred_model_std, y_trues

def plot_predictions(y_true, mean, std, save_path_prefix):
    os.makedirs(os.path.dirname(save_path_prefix), exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.errorbar(y_true, mean, yerr=std, fmt='o', ecolor='lightgray', alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--')
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

def run_ensemble_inference():
    print("Loading test dataset...")
    test_dataset = load_test_data(TEST_DIR, TEST_CSV, batch_size=BATCH_SIZE)

    model_files = sorted([f for f in os.listdir(MODELS_DIR) if f.endswith(".keras")])
    ensemble_dict = {}

    for file in model_files:
        match = re.match(r"(ensemble\d+percent)_model\d+\.keras", file)
        if match:
            ensemble_name = match.group(1)
            ensemble_dict.setdefault(ensemble_name, []).append(file)

    for ensemble_name, model_list in ensemble_dict.items():
        print(f"\nProcessing ensemble: {ensemble_name} with {len(model_list)} models")
        models = [
            tf.keras.models.load_model(os.path.join(MODELS_DIR, f), compile=False)
            for f in model_list
        ]

        pred_mean, aleatoric, epistemic, predictive, pred_model_std, y_true = mc_inference(
            models, test_dataset, n_samples=MC_SAMPLES
        )

        save_path = os.path.join(SAVE_RESULTS_DIR, f"{ensemble_name}_mc_predictions.csv")
        df = pd.DataFrame({
            "y_true": y_true,
            "mean_prediction": pred_mean,
            "aleatoric_uncertainty": aleatoric,
            "epistemic_uncertainty": epistemic,
            "predictive_uncertainty": predictive,
            "model_predicted_std": pred_model_std
        })
        df.to_csv(save_path, index=False)
        print(f"Saved results: {save_path}")

        plot_prefix = os.path.join(SAVE_RESULTS_DIR, ensemble_name)
        plot_predictions(y_true, pred_mean, pred_model_std, plot_prefix)

if __name__ == "__main__":
    run_ensemble_inference()
