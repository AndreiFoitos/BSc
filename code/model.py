import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D

import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D

@tf.keras.utils.register_keras_serializable()
class AgeEstimationModel(tf.keras.Model):
    def __init__(self, input_shape=(224, 224, 3), dropout_rate=0.5, **kwargs):
        super(AgeEstimationModel, self).__init__(**kwargs)
        self.base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
        self.base_model.trainable = False

        self.global_avg_pool = GlobalAveragePooling2D()
        self.dense1 = Dense(512, activation='relu')
        self.dropout = Dropout(dropout_rate)

        # Two separate output layers
        self.age_avg_output = Dense(1, activation='linear', name='apparent_age_avg')
        self.age_std_output = Dense(1, activation='linear', name='apparent_age_std')

    def call(self, inputs):
        x = self.base_model(inputs, training=False)
        x = self.global_avg_pool(x)
        x = self.dense1(x)
        x = self.dropout(x)

        return {
            'apparent_age_avg': self.age_avg_output(x),
            'apparent_age_std': self.age_std_output(x)
        }

    def train(self, train_data, valid_data, epochs=20, train_steps=1000, valid_steps=200):
        self.compile(
            optimizer='adam',
            loss={'apparent_age_avg': 'mse', 'apparent_age_std': 'mse'},  # Only using MSE
            metrics={}  # No MAE
        )
        history = self.fit(train_data, validation_data=valid_data, epochs=epochs,
                           steps_per_epoch=train_steps, validation_steps=valid_steps)
        return history


    def get_config(self):
        config = super(AgeEstimationModel, self).get_config()
        config.update({
            "input_shape": self.base_model.input_shape[1:],
            "dropout_rate": self.dropout.rate
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(input_shape=config['input_shape'], dropout_rate=config['dropout_rate'], name=config['name'])
