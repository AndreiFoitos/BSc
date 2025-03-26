import tensorflow as tf
import matplotlib.pyplot as plt
from model import AgeEstimationModel
from data_loader import AgePredictionDataLoader
import os

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

train_data = train_loader.get_dataset().repeat()
valid_data = valid_loader.get_dataset().repeat()

print("Dataset loaded.")

# Initialize Model
print("Initializing model...")
model = AgeEstimationModel()
print("Model initialized.")

# Train Model
print("Training model...")
history = model.train(train_data, valid_data, epochs=20, train_steps=130, valid_steps=50)

# Save Model
model.save("age_estimation_model.keras")
print("Model training complete and saved as age_estimation_model.keras")

# Plot Learning Curves
def plot_learning_curves(history):
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    
    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(history['mae'], label='Train MAE')
    plt.plot(history['val_mae'], label='Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.title('Mean Absolute Error (MAE)')
    plt.legend()
    
    plt.show()

# Extract history and print final metrics
if hasattr(history, 'history'):
    history_dict = history.history
    plot_learning_curves(history_dict)
    
    print("Final Training MAE:", history_dict['mae'][-1])
    print("Final Validation MAE:", history_dict['val_mae'][-1])
    print("Final Training MSE:", history_dict['mse'][-1])
    print("Final Validation MSE:", history_dict['val_mse'][-1])
else:
    print("History object does not contain history dictionary.")
