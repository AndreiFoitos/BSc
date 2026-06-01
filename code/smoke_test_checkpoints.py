"""Smoke test: load one checkpoint of each method and run a forward pass.

Run BEFORE the full UTKFace eval to catch model-loading issues (custom layer
registration, tfp imports, Flipout serialization, etc.) early.
"""
import os
import numpy as np
import tensorflow as tf

# Importing model.py registers AgeEstimationModel + DropConnectDense via the
# @register_keras_serializable decorator. Without this, load_model may fail.
import model  # noqa: F401

MODELS_DIR = "trained_models_by_fraction"
TARGETS = [
    "dropconnect100percent.keras",
    "flipout100percent.keras",
    "ensemble100percent_model1.keras",
]

def main():
    dummy = np.random.rand(2, 224, 224, 3).astype(np.float32)
    for name in TARGETS:
        path = os.path.join(MODELS_DIR, name)
        if not os.path.exists(path):
            print(f"MISSING: {path}")
            continue
        print(f"\nLoading {name} ...")
        try:
            m = tf.keras.models.load_model(path, compile=False)
        except Exception as e:
            print(f"  LOAD FAILED: {type(e).__name__}: {e}")
            continue
        try:
            preds = m(dummy, training=True)
            mean = preds["apparent_age_avg"].numpy().flatten()
            std = preds["apparent_age_std"].numpy().flatten()
            print(f"  OK. mean shape={mean.shape}, std shape={std.shape}")
            print(f"  example mean={mean.tolist()}, std={std.tolist()}")
        except Exception as e:
            print(f"  FORWARD-PASS FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
