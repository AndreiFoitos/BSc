import tensorflow as tf
from model import AgeEstimationModel
from data_loader import AgePredictionDataLoader
import os

# 
# Paths
BASE_DIR = "appa-real-release/appa-real-release"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VALID_DIR = os.path.join(BASE_DIR, "valid")
TRAIN_CSV = os.path.join(BASE_DIR, "gt_avg_train.csv")
VALID_CSV = os.path.join(BASE_DIR, "gt_avg_valid.csv")

# Load Data
print("Loading dataset...")
train_loader = AgePredictionDataLoader(TRAIN_DIR, TRAIN_CSV, batch_size=32)
valid_loader = AgePredictionDataLoader(VALID_DIR, VALID_CSV, batch_size=32)

train_data = train_loader.get_dataset()
valid_data = valid_loader.get_dataset()

print("Dataset loaded.")

# Initialize Model
print("Initializing model...")
model = AgeEstimationModel()
print("Model initialized.")

# Train Model
print("Training model...")
history = model.train(train_data, valid_data, epochs=20, train_steps=128, valid_steps=32)

# Save Model
model.save("age_estimation_model.h5")
print("Model training complete and saved as age_estimation_model.h5")
