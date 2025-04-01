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
        # Data augmentation
        return ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.2,
            zoom_range=[0.8, 1.2],
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            rescale=1.0 / 255.0  # Rescale images to [0, 1] range
        )

    def _load_labels(self):
        # Load the labels CSV and return a dictionary mapping image file names to age mean and std
        df = pd.read_csv(self.labels_csv)
        labels_dict = {row["file_name"]: (row["apparent_age_avg"], row["apparent_age_std"]) for _, row in df.iterrows()}
        return labels_dict

    def _custom_generator(self):
        # Custom generator to load images and corresponding labels
        image_files = os.listdir(self.data_dir)
        for img_name in image_files:
            if img_name.endswith("_face.jpg"):  # Filter image files that have '_face.jpg'
                csv_name = img_name.replace(".jpg_face.jpg", ".jpg")  # Match CSV file naming convention
                if csv_name in self.labels:
                    img_path = os.path.join(self.data_dir, img_name)
                    if not os.path.exists(img_path):
                        continue
                    try:
                        # Load and process the image
                        img = tf.keras.preprocessing.image.load_img(img_path, target_size=self.target_size)
                        img_array = tf.keras.preprocessing.image.img_to_array(img)
                        img_array = self.datagen.random_transform(img_array)  # Apply augmentation
                        img_array = img_array / 255.0  # Normalize image to [0, 1]
                        
                        # Get labels: apparent_age_avg and apparent_age_std
                        label_avg, label_std = self.labels[csv_name]
                        
                        # Yield the image and the corresponding label (avg, std)
                        yield img_array, tf.convert_to_tensor([label_avg, label_std], dtype=tf.float32)

                    except Exception as e:
                        print(f"Error processing {img_name}: {e}")
                        pass

    def get_dataset(self):
        # Create a tf.data.Dataset from the generator
        dataset = tf.data.Dataset.from_generator(
            self._custom_generator,
            output_signature=(
                tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(2,), dtype=tf.float32)  # Match single tensor output
            )

        )
        
        # Batch the dataset
        dataset = dataset.batch(self.batch_size)
        return dataset
