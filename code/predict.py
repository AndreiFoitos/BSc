import tensorflow as tf
from model import AgeEstimationModel
from data_loader import AgePredictionDataLoader
import os
import numpy as np

# Paths
MODEL_PATH = "age_estimation_model.h5"
TEST_IMAGE_PATH = "appa-real-release/appa-real-release/test/005627.jpg_face.jpg"  # Path to the image or data you want to predict

# Register the custom model class so Keras can use it when loading the model
print("Loading model...")
with tf.keras.utils.custom_object_scope({'AgeEstimationModel': AgeEstimationModel}):
    model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.")

# Load and preprocess the image
img = image.load_img(TEST_IMAGE_PATH, target_size=(224, 224))  # Adjust target_size to match model input
img_array = image.img_to_array(img) / 255.0  # Normalize if your model was trained with normalized images
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Make the prediction
predicted_age = model.predict(img_array)

# Output the predicted age
print("Predicted Age:", predicted_age[0])