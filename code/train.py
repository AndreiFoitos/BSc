import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from model import UncertaintyAgeEstimationModel
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

def build_model():
    learning_rate = 1e-4  # Set a fixed learning rate
    dropout_rate = 0.5  # Set a fixed dropout rate
    
    model = UncertaintyAgeEstimationModel(dropout_rate=dropout_rate)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=model.custom_loss)

    return model

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


# Train final model
model = build_model()
sample_input = tf.random.normal((1, 224, 224, 3))
sample_output = model(sample_input)
print(sample_output.shape)  # Should be (1,2)
sample_img, sample_label = next(iter(train_loader.get_dataset()))
print(sample_label.shape)  # Should print (batch_size, 2)

history = model.fit(train_data, validation_data=valid_data, epochs=20, steps_per_epoch=128, validation_steps=50, callbacks=[early_stopping])

model.save("uncertainty_age_estimation_model.keras")
print("Model training complete and saved.")


# Plot training history
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
