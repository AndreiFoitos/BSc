import os
import json
import random
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import tensorflow as tf

from model import AgeEstimationModel
from data_loader import AgePredictionDataLoader

SEED = 36
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


use_laptop=True
if use_laptop:
    TRAIN_DIR = "C:/Users/Andrei/Documents/GitHub/BSc/appa-real-release/appa-real-release/train"
    VALID_DIR = "C:/Users/Andrei/Documents/GitHub/BSc/appa-real-release/appa-real-release/valid"
    TRAIN_CSV = "C:/Users/Andrei/Documents/GitHub/BSc/appa-real-release/appa-real-release/gt_avg_train.csv"
    VALID_CSV = "C:/Users/Andrei/Documents/GitHub/BSc/appa-real-release/appa-real-release/gt_avg_valid.csv"
else:
    TRAIN_DIR = "C:/Users/Andrei/OneDrive/Documents/GitHub/BSc/appa-real-release/appa-real-release/train"
    VALID_DIR = "C:/Users/Andrei/OneDrive/Documents/GitHub/BSc/appa-real-release/appa-real-release/valid"
    TRAIN_CSV = "C:/Users/Andrei/OneDrive/Documents/GitHub/BSc/appa-real-release/appa-real-release/gt_avg_train.csv"
    VALID_CSV = "C:/Users/Andrei/OneDrive/Documents/GitHub/BSc/appa-real-release/appa-real-release/gt_avg_valid.csv"

BATCH_SIZE = 32
DROPOUT_RATE = 0.3

DATA_FRACTIONS = [0.25, 0.5, 0.75, 1.00]
NUM_ENSEMBLE_MODELS = 3

OUTPUT_DIR = "trained_models_by_fraction"
PLOTS_DIR = "training_plots"
LOGS_DIR = "training_logs"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


def plot_training_history_combined(histories, tag):
    sns.set(style="whitegrid")
    phase_names = ["Phase 1", "Phase 2", "Fine-tune"]
    phase_linestyles = {"Phase 1": "-", "Phase 2": "--", "Fine-tune": ":"}
    phase_markers = {"Phase 1": "o", "Phase 2": "s", "Fine-tune": "^"}

    all_metrics = set()
    for hist in histories:
        all_metrics.update(hist.history.keys())

    color_palette = sns.color_palette("tab10", n_colors=len(all_metrics))
    color_map = {metric: color_palette[i] for i, metric in enumerate(sorted(all_metrics))}

    metrics_combined = {"loss": [], "val_loss": []}
    other_metrics = {}
    total_epochs = 0

    for idx, hist in enumerate(histories):
        phase = phase_names[idx]
        num_epochs = len(hist.history.get("loss", []))

        for i in range(num_epochs):
            metrics_combined["loss"].append((total_epochs + i + 1, hist.history["loss"][i], phase))
            if "val_loss" in hist.history:
                metrics_combined["val_loss"].append((total_epochs + i + 1, hist.history["val_loss"][i], phase))

        for metric, values in hist.history.items():
            if "loss" in metric:
                continue
            if metric not in other_metrics:
                other_metrics[metric] = []
            for i, val in enumerate(values):
                other_metrics[metric].append((total_epochs + i + 1, val, phase))

        total_epochs += num_epochs

    def save_plot_with_legend_outside(path):
        box = plt.gca().get_position()
        plt.gca().set_position([box.x0, box.y0, box.width * 0.75, box.height])
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig(path, bbox_inches="tight")
        plt.close()

    plt.figure(figsize=(10, 6))
    for phase in phase_names:
        linestyle = phase_linestyles[phase]
        marker = phase_markers[phase]

        epochs = [x[0] for x in metrics_combined["loss"] if x[2] == phase]
        losses = [x[1] for x in metrics_combined["loss"] if x[2] == phase]
        plt.plot(epochs, losses, label=f"{phase} - Train", color=color_map["loss"],
                 linestyle=linestyle, marker=marker)

        val_epochs = [x[0] for x in metrics_combined["val_loss"] if x[2] == phase]
        val_losses = [x[1] for x in metrics_combined["val_loss"] if x[2] == phase]
        if val_epochs:
            plt.plot(val_epochs, val_losses, label=f"{phase} - Val", color=color_map["val_loss"],
                     linestyle=linestyle, marker=marker)

    plt.title(f"Loss Across Training Phases - {tag}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    save_plot_with_legend_outside(os.path.join(PLOTS_DIR, f"{tag}_loss_per_phase.png"))

    plt.figure(figsize=(10, 6))
    for metric, records in other_metrics.items():
        for phase in phase_names:
            phase_data = [(e, v) for e, v, p in records if p == phase]
            if phase_data:
                epochs, values = zip(*phase_data)
                plt.plot(epochs, values, label=f"{phase} - {metric}",
                         color=color_map[metric],
                         linestyle=phase_linestyles[phase],
                         marker=phase_markers[phase])

    plt.title(f"Metrics Across Phases - {tag}")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    save_plot_with_legend_outside(os.path.join(PLOTS_DIR, f"{tag}_metrics_per_phase.png"))

    plt.figure(figsize=(10, 6))
    epochs_full = [x[0] for x in metrics_combined["loss"]]
    losses_full = [x[1] for x in metrics_combined["loss"]]
    plt.plot(epochs_full, losses_full, label="Train Loss", color=color_map["loss"], linestyle="-")

    if metrics_combined["val_loss"]:
        val_epochs_full = [x[0] for x in metrics_combined["val_loss"]]
        val_losses_full = [x[1] for x in metrics_combined["val_loss"]]
        plt.plot(val_epochs_full, val_losses_full, label="Val Loss", color=color_map["val_loss"], linestyle="--")

    plt.title(f"Full Loss Progression - {tag}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    save_plot_with_legend_outside(os.path.join(PLOTS_DIR, f"{tag}_loss_full.png"))


def save_history_log(history, tag):
    history_path = os.path.join(LOGS_DIR, f"{tag}_history.json")
    with open(history_path, "w") as f:
        json.dump(history.history, f, indent=2)


def train_and_save(model_type, fraction, model_index=None):
    suffix = f"{int(fraction * 100)}percent"
    tag = f"{model_type}{suffix}" if model_index is None else f"ensemble{suffix}_model{model_index+1}"
    print(f"\n[Training] {tag}")


    train_loader = AgePredictionDataLoader(TRAIN_DIR, TRAIN_CSV, batch_size=BATCH_SIZE, data_fraction=fraction)
    valid_loader = AgePredictionDataLoader(VALID_DIR, VALID_CSV, batch_size=BATCH_SIZE, data_fraction=fraction)
    train_data = train_loader.get_dataset(shuffle=True, repeat=True)
    valid_data = valid_loader.get_dataset(shuffle=False, repeat=True)
    TRAIN_STEPS = (train_loader.get_sample_count() // BATCH_SIZE) + 1

    VALID_STEPS = (valid_loader.get_sample_count() // BATCH_SIZE) + 1

    base_epochs = 100
    epochs_phase1 = int(base_epochs // fraction)
    epochs_phase2 = int(base_epochs // fraction)
    epochs_finetune = int((base_epochs // fraction) * 0.25)

    use_flipout = (model_type == "flipout")
    use_dropconnect = (model_type == "dropconnect")

    model = AgeEstimationModel(
        dropout_rate=DROPOUT_RATE,
        use_flipout=use_flipout,
        use_dropconnect=use_dropconnect,
        dropconnect_rate=0.3
    )

    history_phase1 = model.train_single_task(train_data, valid_data, epochs_phase1, TRAIN_STEPS, VALID_STEPS)
    save_history_log(history_phase1, f"{tag}_phase1")

    history_phase2 = model.train_multitask(train_data, valid_data, epochs_phase2, TRAIN_STEPS, VALID_STEPS)
    save_history_log(history_phase2, f"{tag}_phase2")

    history_fine = model.fine_tune(train_data, valid_data, epochs_finetune, TRAIN_STEPS, VALID_STEPS)
    save_history_log(history_fine, f"{tag}_finetune")

    plot_training_history_combined([history_phase1, history_phase2, history_fine], tag)

    file_name = f"{tag}.keras"
    save_path = os.path.join(OUTPUT_DIR, file_name)
    model.save(save_path)
    print(f"Saved model: {file_name}")



"""for fraction in DATA_FRACTIONS:
    train_and_save("dropconnect", fraction)
    train_and_save("flipout", fraction)"""


for fraction in DATA_FRACTIONS:
    for i in range(NUM_ENSEMBLE_MODELS):
        # Skip training for ensemble25percent_model1 (already trained)
        if fraction == 0.25 and i == 0:
            continue
        train_and_save("ensemble", fraction, model_index=i)



def generate_training_report(log_dir=LOGS_DIR, output_path="training_summary.csv"):
    rows = []
    for file in os.listdir(log_dir):
        if file.endswith("_history.json"):
            full_path = os.path.join(log_dir, file)
            with open(full_path, "r") as f:
                history = json.load(f)

            tag_parts = file.replace("history.json", "").split("")
            phase = tag_parts[-1]
            model_type = tag_parts[0]
            data_fraction = tag_parts[1]
            model_index = tag_parts[3] if "ensemble" in file else None

            for metric, values in history.items():
                if len(values) > 0:
                    rows.append({
                        "model_type": model_type,
                        "data_fraction": data_fraction,
                        "phase": phase,
                        "metric": metric,
                        "value": values[-1],
                        "model_index": model_index
                    })

    df = pd.DataFrame(rows)
    df.sort_values(by=["model_type", "data_fraction", "phase", "metric"], inplace=True)
    df.to_csv(output_path, index=False)
    print(f"Report saved to {output_path}")


generate_training_report()