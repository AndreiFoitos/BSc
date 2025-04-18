import tensorflow as tf
import matplotlib.pyplot as plt
import os
from model import AgeEstimationModel
from data_loader import AgePredictionDataLoader

# ========== Paths & Constants ==========
BASE_DIR = "appa-real-release/appa-real-release"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VALID_DIR = os.path.join(BASE_DIR, "valid")
TRAIN_CSV = os.path.join(BASE_DIR, "gt_avg_train.csv")
VALID_CSV = os.path.join(BASE_DIR, "gt_avg_valid.csv")

BATCH_SIZE = 32
EPOCHS_PHASE1 = 100
EPOCHS_PHASE2 = 100
EPOCHS_FINE_TUNE = 20
TRAIN_STEPS = 128
VALID_STEPS = 50
DROPOUT_RATE = 0.3

# ========== Load Data ==========
train_loader = AgePredictionDataLoader(TRAIN_DIR, TRAIN_CSV, batch_size=BATCH_SIZE, num_samples=4113, bins=10,visualize=True)
valid_loader = AgePredictionDataLoader(VALID_DIR, VALID_CSV, batch_size=BATCH_SIZE,visualize=True)

train_data = train_loader.get_dataset()
valid_data = valid_loader.get_dataset(shuffle=False, repeat=False)

# ========== Initialize Model ==========
model = AgeEstimationModel(dropout_rate=DROPOUT_RATE)
print("Model initialized.")

# ========== Phase 1: Single-task Training ==========
print("\n--- Phase 1: Training only average age (regression) ---")
model.base_model.trainable = False
history1 = model.train_single_task(
    train_data, valid_data,
    epochs=EPOCHS_PHASE1,
    train_steps=TRAIN_STEPS,
    valid_steps=VALID_STEPS
)

# ========== Phase 2: Multi-task with Uncertainty ==========
print("\n--- Phase 2: Multi-task (average + stddev) training ---")
history2 = model.train_multitask(
    train_data, valid_data,
    epochs=EPOCHS_PHASE2,
    train_steps=TRAIN_STEPS,
    valid_steps=VALID_STEPS
)

# ========== Phase 3: Fine-tuning ==========
print("\n--- Phase 3: Fine-tuning base model + heads ---")
fine_tune_history = model.fine_tune(
    train_data, valid_data,
    epochs=EPOCHS_FINE_TUNE,
    train_steps=TRAIN_STEPS,
    valid_steps=VALID_STEPS
)

# ========== Save Model ==========
model.save("age_estimation_model_two_phase.keras")
print("\nTraining complete. Model saved as 'age_estimation_model_two_phase.keras'.")

# ========== Visualization ==========
def plot_history(history, phase):
    plt.figure(figsize=(12, 5))
    for key in history.history:
        linestyle = "--" if "val" in key else "-"
        plt.plot(history.history[key], label=f"{key}", linestyle=linestyle)
    plt.title(f"{phase} Training Metrics")
    plt.xlabel("Epochs")
    plt.ylabel("Loss / Metric")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_history(history1, "Phase 1: Avg Only")
plot_history(history2, "Phase 2: Multi-task")
plot_history(fine_tune_history, "Phase 3: Fine-tuning")
