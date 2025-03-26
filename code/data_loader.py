import tensorflow as tf
import pandas as pd
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class AgePredictionDataLoader:
    def __init__(self, data_dir, labels_csv, batch_size=32, target_size=(224, 224)):
        self.data_dir = data_dir
        self.labels_csv = labels_csv
        self.batch_size = batch_size
        self.target_size = target_size
        self.labels = self._load_labels()
        self.datagen = self._create_datagen()

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

        # Keep original filenames from CSV (without "_face")
        labels_dict = dict(zip(df["file_name"], df["apparent_age_avg"]))
        return labels_dict

    def _custom_generator(self):
        num_loaded = 0  # Debugging counter
        image_files = os.listdir(self.data_dir)  # List all files in directory

        for img_name in image_files:
            if img_name.endswith("_face.jpg"):  # ✅ Only process _face images
                csv_name = img_name.replace(".jpg_face.jpg", ".jpg")

                if csv_name in self.labels:
                    img_path = os.path.join(self.data_dir, img_name)

                    # Debug: Confirm if image exists
                    if not os.path.exists(img_path):
                        continue

                    try:
                        img = tf.keras.preprocessing.image.load_img(img_path, target_size=self.target_size)
                        img_array = tf.keras.preprocessing.image.img_to_array(img)
                        img_array = self.datagen.random_transform(img_array)
                        img_array = img_array / 255.0  # Normalize
                        label = self.labels[csv_name]  # ✅ Match label using original CSV name

                        num_loaded += 1
                        if num_loaded % 10 == 0:
                            pass  # Removed the debug print statement

                        yield img_array, label

                    except Exception as e:
                        pass  # Removed the error print statement

    def get_dataset(self):
        dataset = tf.data.Dataset.from_generator(
            self._custom_generator,
            output_signature=(
                tf.TensorSpec(shape=(self.target_size[0], self.target_size[1], 3), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.float32)
            )
        ).batch(self.batch_size)

        return dataset
