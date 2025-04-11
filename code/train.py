import tensorflow as tf
import matplotlib.pyplot as plt
import os
from model import AgeEstimationModel, EnsembleAgeEstimator
from data_loader import AgePredictionDataLoader

# ==== Paths ====
BASE_DIR = "appa-real-release/appa-real-release"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VALID_DIR = os.path.join(BASE_DIR, "valid")
TRAIN_CSV = os.path.join(BASE_DIR, "gt_avg_train.csv")
VALID_CSV = os.path.join(BASE_DIR, "gt_avg_valid.csv")

# ==== Data ====
train_loader = AgePredictionDataLoader(TRAIN_DIR, TRAIN_CSV, batch_size=32)
valid_loader = AgePredictionDataLoader(VALID_DIR, VALID_CSV, batch_size=32)

train_data = train_loader.get_dataset().repeat()
valid_data = valid_loader.get_dataset().repeat()


print("Dataset loaded.")

# ==== Configuration ====
USE_ENSEMBLE = False
NUM_MODELS = 5
USE_FLIPOUT = False
USE_DROPCONNECT = False
DROPCONNECT_RATE = 0.2
DROPOUT_RATE = 0.3

EPOCHS = 30
FINE_TUNE_EPOCHS = 10
TRAIN_STEPS = 130
VALID_STEPS = 50

# ==== Plotting ====
def plot_learning_curves(history, title_prefix=''):
    plt.figure(figsize=(10, 5))
    plt.plot(history['loss'], label='Train Loss (Gaussian NLL)')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Val Loss (Gaussian NLL)')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{title_prefix} Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ==== Training ====
if USE_ENSEMBLE:
    print(f"Initializing Ensemble with {NUM_MODELS} models (Flipout: {USE_FLIPOUT}, DropConnect: {USE_DROPCONNECT})")

    ensemble = EnsembleAgeEstimator(
        num_models=NUM_MODELS,
        dropout_rate=DROPOUT_RATE,
        use_flipout=USE_FLIPOUT,
        use_dropconnect=USE_DROPCONNECT,
        dropconnect_rate=DROPCONNECT_RATE
    )

    ensemble.compile_all()
    histories = ensemble.fit_all(train_data, valid_data,
                                 epochs=EPOCHS, train_steps=TRAIN_STEPS, valid_steps=VALID_STEPS)

    for i, history in enumerate(histories):
        if hasattr(history, 'history'):
            plot_learning_curves(history.history, title_prefix=f'Model {i+1}')
            final_loss = history.history['val_loss'][-1]
            print(f"Model {i+1} Final Val Gaussian NLL Loss: {final_loss:.4f}")

    print("Ensemble training complete.")

    for i, model in enumerate(ensemble.models):
        model.save(f"ensemble_model_{i+1}.keras")
        print(f"Saved ensemble_model_{i+1}.keras")

else:
    print("Initializing Single Age Estimation Model")
    model = AgeEstimationModel(
        dropout_rate=DROPOUT_RATE,
        use_flipout=USE_FLIPOUT,
        use_dropconnect=USE_DROPCONNECT,
        dropconnect_rate=DROPCONNECT_RATE
    )

    print("Training started...")
    history = model.train(train_data, valid_data, epochs=EPOCHS,
                          train_steps=TRAIN_STEPS, valid_steps=VALID_STEPS)

    print("Fine-tuning...")
    fine_tune_history = model.fine_tune(train_data, valid_data,
                                        epochs=FINE_TUNE_EPOCHS,
                                        train_steps=TRAIN_STEPS,
                                        valid_steps=VALID_STEPS)

    model.save("age_estimation_model.keras")
    print("Model saved to age_estimation_model.keras")

# Plot training loss
if hasattr(history, 'history'):
    plot_learning_curves(history.history, title_prefix='Initial Training')

# Plot fine-tuning loss
if hasattr(fine_tune_history, 'history'):
    plot_learning_curves(fine_tune_history.history, title_prefix='Fine-Tuning')

final_loss = history.history['val_loss'][-1]
print(f"Final Val Gaussian NLL Loss: {final_loss:.4f}")
