import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import os

import random
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
from tensorflow.keras.applications.densenet import preprocess_input #type: ignore


SEED = 36
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

class AgePredictionDataLoader:
    def __init__(self, data_dir, labels_csv, batch_size=32, target_size=(224, 224),
                 data_fraction=1.0, visualize=False, stratify_bins=10):
        self.data_dir = data_dir
        self.labels_csv = labels_csv
        self.batch_size = batch_size
        self.target_size = target_size
        self.data_fraction = data_fraction
        self.stratify_bins = stratify_bins
        self.visualize = visualize

        self.labels_df = self._load_labels()
        self.labels_df = self._sample_labels(self.labels_df)
        self.labels = self._build_label_dict()
        self.datagen = self._create_datagen()

        if self.visualize:
            self._visualize_distribution()

    def _create_datagen(self):
        return ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.2,
            zoom_range=[0.8, 1.2],
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
        )

    def _load_labels(self):
        return pd.read_csv(self.labels_csv)

    def _sample_labels(self, df):
        if self.data_fraction >= 1.0:
            return df

        df['age_bin'] = pd.qcut(df['apparent_age_avg'], q=self.stratify_bins, duplicates='drop')
        sampled_df = df.groupby('age_bin', group_keys=False).apply(
            lambda x: x.sample(frac=self.data_fraction, random_state=42)
        )
        return sampled_df.drop(columns='age_bin')

    def _build_label_dict(self):
        return dict(zip(
            self.labels_df["file_name"],
            zip(self.labels_df["apparent_age_avg"], self.labels_df["apparent_age_std"])
        ))

    def _visualize_distribution(self):
        plt.figure(figsize=(8, 5))
        plt.hist(self.labels_df['apparent_age_avg'], bins=20, color='skyblue', edgecolor='black')
        plt.title("Apparent Age Distribution (Sampled)" if self.data_fraction < 1.0 else "Apparent Age Distribution")
        plt.xlabel("Apparent Age (Average)")
        plt.ylabel("Count")
        plt.grid(True)
        plt.show()

    def _custom_generator(self):
        used = set()
        image_files = [f for f in os.listdir(self.data_dir) if f.endswith("_face.jpg")]

        for img_name in image_files:
            csv_name = img_name.replace("_face.jpg", "")
            if csv_name in self.labels and csv_name not in used:
                used.add(csv_name)
                img_path = os.path.join(self.data_dir, img_name)

                if not os.path.exists(img_path):
                    continue

                try:
                    img = tf.keras.preprocessing.image.load_img(img_path, target_size=self.target_size)
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    img_array = self.datagen.random_transform(img_array)
                    img_array = preprocess_input(img_array)
                    age_label, std_label = self.labels[csv_name]

                    yield img_array, {
                        "apparent_age_avg": tf.convert_to_tensor(age_label, dtype=tf.float32),
                        "apparent_age_std": tf.convert_to_tensor(std_label, dtype=tf.float32)
                    }
                except Exception as e:
                    print(f"[ERROR] Failed to process image {img_path}: {e}")
                    continue

    def get_dataset(self, shuffle=True, repeat=True, prefetch=True):
        dataset = tf.data.Dataset.from_generator(
            self._custom_generator,
            output_signature=(
                tf.TensorSpec(shape=(self.target_size[0], self.target_size[1], 3), dtype=tf.float32),
                {
                    "apparent_age_avg": tf.TensorSpec(shape=(), dtype=tf.float32),
                    "apparent_age_std": tf.TensorSpec(shape=(), dtype=tf.float32)
                }
            )
        )

        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)

        if repeat:
            dataset = dataset.repeat()

        dataset = dataset.batch(self.batch_size, drop_remainder=False)

        if prefetch:
            dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return dataset

    def get_sample_count(self):
        return len(self.labels_df)
