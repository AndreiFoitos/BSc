import tensorflow as tf
import matplotlib.pyplot as plt
from model import AgeEstimationModel
from data_loader import AgePredictionDataLoader
import os

BASE_DIR = "appa-real-release/appa-real-release"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VALID_DIR = os.path.join(BASE_DIR, "valid")
TRAIN_CSV = os.path.join(BASE_DIR, "gt_avg_train.csv")
VALID_CSV = os.path.join(BASE_DIR, "gt_avg_valid.csv")

train_loader = AgePredictionDataLoader(TRAIN_DIR, TRAIN_CSV, batch_size=32)
valid_loader = AgePredictionDataLoader(VALID_DIR, VALID_CSV, batch_size=32)

train_data = train_loader.get_dataset().repeat()
valid_data = valid_loader.get_dataset().repeat()

print("Dataset loaded.")

model = AgeEstimationModel()
print("Model initialized.")

history = model.train(train_data, valid_data, epochs=100, train_steps=130, valid_steps=50)

# Fine-tune the model after initial training
fine_tune_history = model.fine_tune(train_data, valid_data, epochs=10, train_steps=130, valid_steps=50)

model.save("age_estimation_model.keras")
print("Model training complete and saved as age_estimation_model.keras")

def plot_learning_curves(history):
    plt.figure(figsize=(10, 5))

    plt.plot(history['apparent_age_avg_loss'], label='Train Loss (Avg)')
    plt.plot(history['val_apparent_age_avg_loss'], label='Validation Loss (Avg)')
    plt.plot(history['apparent_age_std_loss'], label='Train Loss (Std)')
    plt.plot(history['val_apparent_age_std_loss'], label='Validation Loss (Std)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title('Loss Curves (MSE)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if hasattr(history, 'history'):
    history_dict = history.history
    plot_learning_curves(history_dict)

    print("Final Train MSE (Avg):", history_dict['apparent_age_avg_loss'][-1])
    print("Final Val MSE (Avg):", history_dict['val_apparent_age_avg_loss'][-1])
    print("Final Train MSE (Std):", history_dict['apparent_age_std_loss'][-1])
    print("Final Val MSE (Std):", history_dict['val_apparent_age_std_loss'][-1])
else:
    print("History object does not contain history dictionary.")
