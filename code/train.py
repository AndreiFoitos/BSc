import tensorflow as tf
import matplotlib.pyplot as plt
import os
from model import AgeEstimationModel
from data_loader import AgePredictionDataLoader

# ========== Paths & Constants ==========
BASE_DIR = "C:/Users/Andrei/Documents/GitHub/BSc/appa-real-release/appa-real-release"
TRAIN_DIR = "C:/Users/Andrei/Documents/GitHub/BSc/appa-real-release/appa-real-release/train"
VALID_DIR = "C:/Users/Andrei/Documents/GitHub/BSc/appa-real-release/appa-real-release/valid"
TRAIN_CSV = "C:/Users/Andrei/Documents/GitHub/BSc/appa-real-release/appa-real-release/gt_avg_train.csv"
VALID_CSV = "C:/Users/Andrei/Documents/GitHub/BSc/appa-real-release/appa-real-release/gt_avg_valid.csv"

BATCH_SIZE = 32
EPOCHS_PHASE1 = 100
EPOCHS_PHASE2 = 100
EPOCHS_FINE_TUNE = 20
TRAIN_STEPS = 128
VALID_STEPS = 46
DROPOUT_RATE = 0.3
USE_FLIPOUT=False
USE_DROPCONNECT=True
USE_ENSAMBLE=False

# ========== Load Data ==========
# Training: repeat=True, shuffle=True
train_loader = AgePredictionDataLoader(TRAIN_DIR, TRAIN_CSV, batch_size=32, data_fraction=.1, visualize=True)
train_data = train_loader.get_dataset(shuffle=True, repeat=True)

# Validation: repeat=False, shuffle=False
valid_loader = AgePredictionDataLoader(VALID_DIR, VALID_CSV, batch_size=32, visualize=True)
valid_data = valid_loader.get_dataset(shuffle=False, repeat=False)

print("Train size:", train_loader.get_sample_count())
print("Validation size:", valid_loader.get_sample_count())


if USE_ENSAMBLE:
    NUM_MODELS = 5  # <--- Number of models in the ensemble

    OUTPUT_DIR = "ensemble_models"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    models = []
    histories = []

    
    def plot_history(history, phase, model_index):
        plt.figure(figsize=(10, 5))
        for key in history.history:
            linestyle = "--" if "val" in key else "-"
            plt.plot(history.history[key], label=key, linestyle=linestyle)
        plt.title(f"Model {model_index+1} - {phase}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss / Metric")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"model_{model_index+1}_{phase.lower().replace(' ', '_')}.png"))
        plt.close()

    for i in range(NUM_MODELS):
        print(f"\n================== [Model {i+1}/{NUM_MODELS}] ==================")

        # Alternate between regular, flipout, and dropconnect
        use_flipout = (i % 3 == 0)
        use_dropconnect = (i % 3 == 1)

        model = AgeEstimationModel(
            dropout_rate=DROPOUT_RATE,
            use_flipout=use_flipout,
            use_dropconnect=use_dropconnect,
            dropconnect_rate=0.3
        )

        print(f"Model {i+1} initialized. [Flipout: {use_flipout}] [DropConnect: {use_dropconnect}]")

        # ========== Phase 1 ==========
        print("--- Phase 1: Single-task Training ---")
        model.base_model.trainable = False
        hist1 = model.train_single_task(
            train_data, valid_data,
            epochs=EPOCHS_PHASE1,
            train_steps=TRAIN_STEPS,
            valid_steps=VALID_STEPS
        )
        plot_history(hist1, "Phase 1", i)

        # ========== Phase 2 ==========
        print("--- Phase 2: Multi-task Training ---")
        hist2 = model.train_multitask(
            train_data, valid_data,
            epochs=EPOCHS_PHASE2,
            train_steps=TRAIN_STEPS,
            valid_steps=VALID_STEPS
        )
        plot_history(hist2, "Phase 2", i)

        # ========== Phase 3 ==========
        print("--- Phase 3: Fine-tuning ---")
        hist3 = model.fine_tune(
            train_data, valid_data,
            epochs=EPOCHS_FINE_TUNE,
            train_steps=TRAIN_STEPS,
            valid_steps=VALID_STEPS
        )
        plot_history(hist3, "Phase 3", i)

        # ========== Save Model ==========
        filename = f"ensemble_model_{i+1}.keras"
        path = os.path.join(OUTPUT_DIR, filename)
        model.save(path)
        print(f"Model {i+1} saved to {path}")

        print("\nAll ensemble models trained and saved.")
else:

    # ========== Initialize Model ==========
    model = AgeEstimationModel(dropout_rate=DROPOUT_RATE, use_flipout=USE_FLIPOUT, use_dropconnect=USE_DROPCONNECT,dropconnect_rate=0.3)
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
    if USE_DROPCONNECT:
        model.save("age_estimation_model_two_phase_dropconnect.keras")
        print("\nTraining complete. Model saved as 'age_estimation_model_two_phase_dropconnect.keras'.")
    elif USE_FLIPOUT:
        model.save("age_estimation_model_two_phase_flipout.keras")
        print("\nTraining complete. Model saved as 'age_estimation_model_two_phase_flipout.keras'.")
    else:
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
