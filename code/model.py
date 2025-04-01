import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

@tf.keras.utils.register_keras_serializable()
class UncertaintyAgeEstimationModel(Model):
    def __init__(self, dropout_rate=0.5, **kwargs):
        super(UncertaintyAgeEstimationModel, self).__init__(**kwargs)
        self.base_model = tf.keras.applications.ResNet50(include_top=False, input_shape=(224, 224, 3), pooling="avg")
        self.dropout = layers.Dropout(dropout_rate)
        self.age_output = layers.Dense(1, name="age_pred")  # Age prediction
        self.log_var_output = layers.Dense(1, name="log_var_pred")  # Log variance output

    def call(self, inputs):
        x = self.base_model(inputs, training=False)
        x = self.dropout(x)
        age_pred = self.age_output(x)
        log_var_pred = self.log_var_output(x)  
        return tf.concat([age_pred, log_var_pred], axis=1)  # Ensure 2D output

    def get_config(self):
        config = super().get_config()
        config.update({"dropout_rate": self.dropout.rate})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @staticmethod
    def custom_loss(y_true, y_pred):
        y_true = tf.convert_to_tensor(y_true)
        age_true, age_pred = y_true[:, 0], y_pred[:, 0]
        log_var = y_pred[:, 1]  

        mse = tf.square(age_true - age_pred) / (2 * tf.exp(log_var))
        reg = 0.5 * log_var
        return tf.reduce_mean(mse + reg)

    def train(self, train_data, valid_data, epochs=20, train_steps=1000, valid_steps=200):
        self.compile(optimizer='adam', loss=self.custom_loss)
        history = self.fit(train_data, validation_data=valid_data, epochs=epochs,
                           steps_per_epoch=train_steps, validation_steps=valid_steps)
        return history

    def mc_dropout_predict(self, x, num_samples=50):
        preds = np.array([self(x, training=True)[:, 0].numpy() for _ in range(num_samples)])
        mean_prediction = preds.mean(axis=0)
        epistemic_uncertainty = preds.std(axis=0)
        return mean_prediction, epistemic_uncertainty
