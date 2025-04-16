import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class AgePredictionDataLoader:
    def __init__(self, data_dir, labels_csv, batch_size=32, target_size=(224, 224),
                 num_samples=1000, bins=10, visualize=True):
        self.data_dir = data_dir
        self.labels_csv = labels_csv
        self.batch_size = batch_size
        self.target_size = target_size
        self.num_samples = num_samples
        self.bins = bins
        self.visualize = visualize
        self.labels_df = self._load_labels()
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
            rescale=1.0 / 255.0
        )

    def _load_labels(self):
        df = pd.read_csv(self.labels_csv)

        if self.num_samples is None or self.bins is None:
            return df  # no balanced sampling, return all

        df['age_bin'] = pd.cut(df["apparent_age_avg"], bins=self.bins, labels=False)
        per_bin = max(1, self.num_samples // self.bins)
        grouped = df.groupby("age_bin")
        sampled_df = grouped.apply(
            lambda x: x.sample(min(len(x), per_bin), random_state=42)
        ).reset_index(drop=True)

        return sampled_df


    def _build_label_dict(self):
        return dict(zip(
            self.labels_df["file_name"],
            zip(self.labels_df["apparent_age_avg"], self.labels_df["apparent_age_std"])
        ))

    def _visualize_distribution(self):
        plt.figure(figsize=(8, 5))
        plt.hist(self.labels_df['apparent_age_avg'], bins=self.bins, color='skyblue', edgecolor='black')
        plt.title("Sampled Balanced Apparent Age Distribution")
        plt.xlabel("Apparent Age (Average)")
        plt.ylabel("Count")
        plt.grid(True)
        plt.show()

    def _custom_generator(self):
        image_files = os.listdir(self.data_dir)
        used = set()

        for img_name in image_files:
            if img_name.endswith("_face.jpg"):
                csv_name = img_name.replace(".jpg_face.jpg", ".jpg")

                if csv_name in self.labels and csv_name not in used:
                    used.add(csv_name)
                    img_path = os.path.join(self.data_dir, img_name)

                    if not os.path.exists(img_path):
                        continue

                    try:
                        img = tf.keras.preprocessing.image.load_img(img_path, target_size=self.target_size)
                        img_array = tf.keras.preprocessing.image.img_to_array(img)
                        img_array = self.datagen.random_transform(img_array)
                        img_array = img_array / 255.0
                        age_label, std_label = self.labels[csv_name]

                        yield img_array, {
                            "apparent_age_avg": tf.convert_to_tensor(age_label, dtype=tf.float32),
                            "apparent_age_std": tf.convert_to_tensor(std_label, dtype=tf.float32)
                        }

                    except Exception as e:
                        continue

    def get_dataset(self, shuffle=True, repeat=False, prefetch=True):
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
            dataset = dataset.shuffle(buffer_size=self.num_samples)

        if repeat:
            dataset = dataset.repeat()

        dataset = dataset.batch(self.batch_size)

        if prefetch:
            dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return dataset
