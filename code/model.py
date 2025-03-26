import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D

import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D

@tf.keras.utils.register_keras_serializable()  # Correct decorator
class AgeEstimationModel(tf.keras.Model):
    def __init__(self, input_shape=(224, 224, 3), dropout_rate=0.5, **kwargs):
        super(AgeEstimationModel, self).__init__(**kwargs)
        self.base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
        self.base_model.trainable = False  # Freeze the base model

        self.global_avg_pool = GlobalAveragePooling2D()
        self.dense1 = Dense(512, activation='relu')
        self.dropout = Dropout(dropout_rate)
        self.output_layer = Dense(1, activation='linear')  # Regression output (age)

    def call(self, inputs):
        x = self.base_model(inputs, training=False)
        x = self.global_avg_pool(x)
        x = self.dense1(x)
        x = self.dropout(x)
        return self.output_layer(x)

    def get_config(self):
        # Return the config for all arguments needed for the model
        config = super(AgeEstimationModel, self).get_config()
        config.update({
            "input_shape": self.base_model.input_shape[1:],  # Exclude batch dimension
            "dropout_rate": self.dropout.rate
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Custom deserialization
        return cls(input_shape=config['input_shape'], dropout_rate=config['dropout_rate'], name=config['name'])


    def train(self, train_data, valid_data, epochs=20, train_steps=1000, valid_steps=200):
        self.compile(optimizer='adam', loss='mse', metrics=['mae'])
        history = self.fit(train_data, validation_data=valid_data, epochs=epochs,
                           steps_per_epoch=train_steps, validation_steps=valid_steps)
        return history

    def predict_age(self, img_path):
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        img_array = tf.expand_dims(img_array, axis=0)  # Convert to batch format
        return self.predict(img_array)[0][0]  # Return predicted age
