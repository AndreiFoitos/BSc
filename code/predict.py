import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score

# ================= Configuration =================
<<<<<<< HEAD
USE_FLIPOUT = True

USE_DROPCONNECT = False
=======
USE_FLIPOUT = False
USE_DROPCONNECT = True
>>>>>>> 0dc6b61f45adf42a6546daaadf21c43ac71e0849
USE_ENSEMBLE = False  # Set to True to evaluate ensemble

NUM_ENSEMBLE_MODELS = 5
MODEL_BASE_PATH = "ensemble_models"

<<<<<<< HEAD


use_laptop=False
if use_laptop:
    TEST_DIR = "C:/Users/Andrei/Documents/GitHub/BSc/appa-real-release/appa-real-release/test"
    TEST_CSV = "C:/Users/Andrei/Documents/GitHub/BSc/appa-real-release/appa-real-release/gt_avg_test.csv"
else:
    TEST_DIR = "C:/Users/Andrei/OneDrive/Documents/GitHub/BSc/appa-real-release/appa-real-release/test"
    TEST_CSV = "C:/Users/Andrei/OneDrive/Documents/GitHub/BSc/appa-real-release/appa-real-release/gt_avg_test.csv"

MODEL_PATH = "age_estimation_model_two_phase.keras"  # Default single model path

=======
TEST_DIR = "C:/Users/Andrei/Documents/GitHub/BSc/appa-real-release/appa-real-release/test"
TEST_CSV = "C:/Users/Andrei/Documents/GitHub/BSc/appa-real-release/appa-real-release/gt_avg_test.csv"
>>>>>>> 0dc6b61f45adf42a6546daaadf21c43ac71e0849

# ================= Load Test Data =================
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
    dataset = dataset.batch(batch_size)
    return dataset


<<<<<<< HEAD
# ================= Evaluation Functions =================
def evaluate_with_dataloader(model, dataset, return_predictions=False, training_mode=False):
    actual_ages, predicted_ages, actual_stds, predicted_stds, errors = [], [], [], [], []

    for images, labels in dataset:
        preds = model(images, training=training_mode)  # 👈 use training=True for Flipout

        pred_avg = preds["apparent_age_avg"].numpy().flatten()
        pred_std = preds["apparent_age_std"].numpy().flatten()

        actual_avg = labels["apparent_age_avg"].numpy().flatten()
        actual_std = labels["apparent_age_std"].numpy().flatten()

        errors.extend(np.abs(pred_avg - actual_avg))
        actual_ages.extend(actual_avg)
        predicted_ages.extend(pred_avg)
        actual_stds.extend(actual_std)
        predicted_stds.extend(pred_std)

    mse = np.mean(np.square(errors))
    rmse = np.sqrt(mse)
    mae = np.mean(errors)
    r2 = r2_score(actual_ages, predicted_ages)

    print(f"\nEvaluation Results:")
    print(f"MAE : {mae:.2f}")
    print(f"MSE : {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²  : {r2:.3f}")

    plot_results(actual_ages, predicted_ages, predicted_stds)
    plot_residuals(actual_ages, predicted_ages)
    plot_residual_histogram(actual_ages, predicted_ages)
    plot_binned_mse(actual_ages, predicted_ages)
    plot_heatmap(actual_ages, predicted_ages)
    plot_calibration_curve(actual_ages, predicted_ages)

    if return_predictions:
        return predicted_ages, predicted_stds


def export_predictions_to_csv(predicted_ages, predicted_stds, original_csv_path, model_type="default"):
    output_csv_path = f"predictions_{model_type}.csv"
    df = pd.read_csv(original_csv_path)

    min_len = min(len(predicted_ages), len(df))
    df = df.iloc[:min_len]
    df["predicted_age"] = predicted_ages[:min_len]
    df["predicted_std"] = predicted_stds[:min_len]

    df.to_csv(output_csv_path, index=False)
    print(f"\nPredictions saved to {output_csv_path}")


def load_ensemble_model_predictions(dataset, model_paths):
    all_preds_avg, all_preds_std = [], []

    for path in model_paths:
        print(f"Loading {path}...")
        model = tf.keras.models.load_model(path, compile=False)
        preds_avg, preds_std = [], []
=======
# =================== MC Inference ===================
def mc_inference(models, dataset, n_samples=20):
    all_means = []
    all_vars = []

    for _ in range(n_samples):
        means = []
        variances = []
>>>>>>> 0dc6b61f45adf42a6546daaadf21c43ac71e0849

        for images, _ in dataset:
            batch_means = []
            batch_vars = []

            for model in models:
                preds = model(images, training=True)
                mean = preds["apparent_age_avg"].numpy().flatten()
                std = preds["apparent_age_std"].numpy().flatten()
                var = np.square(std)
                batch_means.append(mean)
                batch_vars.append(var)

            mean_avg = np.mean(batch_means, axis=0)
            var_avg = np.mean(batch_vars, axis=0)

            means.append(mean_avg)
            variances.append(var_avg)

        all_means.append(np.concatenate(means))
        all_vars.append(np.concatenate(variances))

    all_means = np.stack(all_means)
    all_vars = np.stack(all_vars)

    pred_mean = np.mean(all_means, axis=0)
    aleatoric = np.mean(all_vars, axis=0)
    epistemic = np.var(all_means, axis=0)
    predictive = aleatoric + epistemic

    return pred_mean, aleatoric, epistemic, predictive


# =================== Visualization ===================
def plot_uncertainty_components(actual, predicted, aleatoric, epistemic, predictive):
    plt.figure(figsize=(10, 6))
    plt.scatter(actual, predicted, c='blue', label="Predicted Ages", alpha=0.6)
    plt.errorbar(actual, predicted, yerr=np.sqrt(aleatoric), fmt='o', color='gray', alpha=0.3, label='Aleatoric Std')
    plt.errorbar(actual, predicted, yerr=np.sqrt(epistemic), fmt='o', color='orange', alpha=0.3, label='Epistemic Std')
    plt.plot([min(actual), max(actual)], [min(actual), max(actual)], 'r--', label="Perfect Prediction")
    plt.xlabel("Actual Age")
    plt.ylabel("Predicted Age")
    plt.title("Age Prediction with Disentangled Uncertainty")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# =================== Main Execution ===================
if __name__ == "__main__":
    test_data = load_test_data(TEST_DIR, TEST_CSV)
    print("✅ Test dataset loaded.")

    models = []

    if USE_ENSEMBLE:
        model_paths = [os.path.join(MODEL_BASE_PATH, f"ensemble_model_{i+1}.keras") for i in range(NUM_ENSEMBLE_MODELS)]
        print(f"📦 Loading ensemble models...")
        for path in model_paths:
            model = tf.keras.models.load_model(path, compile=False)
            models.append(model)
        model_type = "ensemble"
    else:
        if USE_FLIPOUT:
            MODEL_PATH = "age_estimation_model_two_phase_flipout.keras"
            model_type = "flipout"
        elif USE_DROPCONNECT:
            MODEL_PATH = "age_estimation_model_two_phase_dropconnect.keras"
            model_type = "dropconnect"
        else:
            MODEL_PATH = "age_estimation_model_two_phase.keras"
            model_type = "default"

<<<<<<< HEAD
    print(f"Loading model: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("Model loaded.")

    predicted_ages, predicted_stds = evaluate_with_dataloader(model, test_data, return_predictions=True, training_mode=USE_FLIPOUT)
    export_predictions_to_csv(predicted_ages, predicted_stds, TEST_CSV, model_type=model_type)
=======
        print(f"📦 Loading model: {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        models.append(model)

    print("\n🔁 Running Monte Carlo inference for uncertainty disentanglement...")
    pred_mean, aleatoric, epistemic, predictive = mc_inference(models, test_data, n_samples=20)

    df = pd.read_csv(TEST_CSV)
    actual_ages = df["real_age"].astype(np.float32).values
    min_len = min(len(pred_mean), len(actual_ages))

    actual_ages = actual_ages[:min_len]
    pred_mean = pred_mean[:min_len]
    aleatoric = aleatoric[:min_len]
    epistemic = epistemic[:min_len]
    predictive = predictive[:min_len]

    df = df.iloc[:min_len]
    df["predicted_age"] = pred_mean
    df["aleatoric_uncertainty"] = aleatoric
    df["epistemic_uncertainty"] = epistemic
    df["predictive_uncertainty"] = predictive

    out_csv = f"predictions_disentangled_{model_type}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n📁 Predictions with disentangled uncertainty saved to: {out_csv}")

    plot_uncertainty_components(actual_ages, pred_mean, aleatoric, epistemic, predictive)
>>>>>>> 0dc6b61f45adf42a6546daaadf21c43ac71e0849
