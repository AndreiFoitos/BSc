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
    
    predicted_age = model.predict(img_array)[0][0]
    return predicted_age

def load_ground_truth(csv_path):
    """ Load ground truth ages from CSV file """
    df = pd.read_csv(csv_path)
    return dict(zip(df['file_name'], df['apparent_age_avg']))

def predict_and_compare(model, folder_path, csv_path):
    ground_truth = load_ground_truth(csv_path)
    if not os.path.exists(folder_path):
        print("Folder not found!")
        return
    
    actual_ages = []
    predicted_ages = []
    errors = []
    
    filenames = [f for f in os.listdir(folder_path) if f.lower().endswith(('jpg', 'jpeg', 'png')) and f in ground_truth]

    for i, filename in enumerate(filenames):
        img_path = os.path.join(folder_path, filename)
        predicted_age = predict_age(model, img_path)
        actual_age = ground_truth[filename]
        
        error = abs(predicted_age - actual_age)
        errors.append(error)
        actual_ages.append(actual_age)
        predicted_ages.append(predicted_age)

        
        if i < 10:
            print(f"{filename}: Predicted Age = {predicted_age:.2f}, Apparent Age Avg = {actual_age}, Error = {error:.2f}")
    
    mae = np.mean(errors)
    mse = np.mean(np.array(errors) ** 2)

    print(f"\nMean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")

    plot_results(actual_ages, predicted_ages)

def plot_results(actual_ages, predicted_ages):
    """ Plot scatter plot of actual vs predicted ages """
    plt.figure(figsize=(8, 8))
    plt.scatter(actual_ages, predicted_ages, color='blue', alpha=0.6, label="Predictions")
    plt.plot([min(actual_ages), max(actual_ages)], [min(actual_ages), max(actual_ages)], 'r--', label="Perfect Prediction (y=x)")
    
    plt.xlabel("Actual Age")
    plt.ylabel("Predicted Age")
    plt.title("Age Estimation: Predicted vs Actual Age")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    folder_path = "appa-real-release/appa-real-release/test"
    csv_path = "appa-real-release/appa-real-release/gt_avg_test.csv"
    model = load_model()
    predict_and_compare(model, folder_path, csv_path)