import tensorflow as tf
from model import AgeEstimationModel
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

def load_model(model_path="age_estimation_model.keras"):
    print("Loading model...")
    model = tf.keras.models.load_model(model_path, compile=False)
    print("Model loaded successfully.")
    return model

def predict_age(model, img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)

    # Extracting from dictionary correctly
    if isinstance(predictions, dict):
        predicted_age_avg = predictions["apparent_age_avg"][0][0]  # Extract scalar from array
        predicted_age_std = predictions["apparent_age_std"][0][0]  # Extract scalar from array
    else:
        raise ValueError(f"Unexpected model output format: {type(predictions)} with structure {predictions}")

    return predicted_age_avg, predicted_age_std


def load_ground_truth(csv_path):
    """ Load ground truth ages from CSV file """
    df = pd.read_csv(csv_path)
    return {row['file_name']: (row['apparent_age_avg'], row['apparent_age_std']) for _, row in df.iterrows()}

def predict_and_compare(model, folder_path, csv_path):
    ground_truth = load_ground_truth(csv_path)
    if not os.path.exists(folder_path):
        print("Folder not found!")
        return
    
    actual_ages = []
    predicted_ages = []
    predicted_stds = []
    errors = []
    
    filenames = [f for f in os.listdir(folder_path) if f.lower().endswith(('jpg', 'jpeg', 'png')) and f in ground_truth]

    for i, filename in enumerate(filenames):
        img_path = os.path.join(folder_path, filename)
        predicted_age_avg, predicted_age_std = predict_age(model, img_path)
        
        # Unpack ground truth values correctly
        actual_age, actual_std = ground_truth[filename]
        
        error = abs(predicted_age_avg - actual_age)
        errors.append(error)
        actual_ages.append(actual_age)
        predicted_ages.append(predicted_age_avg)
        predicted_stds.append(predicted_age_std)

        if i < 10:
            print(f"{filename}: Predicted Age = {predicted_age_avg:.2f} ± {predicted_age_std:.2f}, "
                  f"Avg Apparent Age = {actual_age:.2f} ± {actual_std:.2f}, Error = {error:.2f}")
    
    mse = np.mean(np.array(errors) ** 2)
    print(f"\nMean Squared Error (MSE): {mse:.2f}")
    plot_results(actual_ages, predicted_ages, predicted_stds)

def plot_results(actual_ages, predicted_ages, predicted_stds):
    """ Plot scatter plot of actual vs predicted ages with error bars """
    plt.figure(figsize=(8, 8))
    plt.errorbar(actual_ages, predicted_ages, yerr=predicted_stds, fmt='o', color='blue', alpha=0.6, label="Predictions with Std Dev")
    plt.plot([min(actual_ages), max(actual_ages)], [min(actual_ages), max(actual_ages)], 'r--', label="Perfect Prediction (y=x)")
    
    plt.xlabel("Actual Age")
    plt.ylabel("Predicted Age")
    plt.title("Age Estimation: Predicted vs Actual Age with Uncertainty")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    folder_path = "appa-real-release/appa-real-release/test"
    csv_path = "appa-real-release/appa-real-release/gt_avg_test.csv"
    model = load_model()
    predict_and_compare(model, folder_path, csv_path)