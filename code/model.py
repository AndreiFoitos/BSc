import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping


class DropConnectDense(Dense):
    def __init__(self, units, dropconnect_rate=0.0, **kwargs):
        super().__init__(units, **kwargs)
        self.dropconnect_rate = dropconnect_rate

    def call(self, inputs, training=None):
        kernel = self.kernel
        if training:
            drop_mask = tf.nn.dropout(tf.ones_like(kernel), rate=self.dropconnect_rate)
            kernel = kernel * drop_mask
        return tf.matmul(inputs, kernel) + self.bias


@tf.keras.utils.register_keras_serializable()
class AgeEstimationModel(tf.keras.Model):
    def __init__(self, input_shape=(224, 224, 3), dropout_rate=0.5,
                 use_flipout=False, use_dropconnect=False, dropconnect_rate=0.2, **kwargs):
        super(AgeEstimationModel, self).__init__(**kwargs)

        self.use_flipout = use_flipout
        self.use_dropconnect = use_dropconnect
        self.dropconnect_rate = dropconnect_rate

        self.base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
        self.base_model.trainable = False

        self.global_avg_pool = GlobalAveragePooling2D()

        self.dense1 = self._get_dense(512)
        self.batch_norm1 = BatchNormalization()
        self.relu1 = LeakyReLU(alpha=0.1)

        self.dense2 = self._get_dense(256)
        self.batch_norm2 = BatchNormalization()
        self.relu2 = LeakyReLU(alpha=0.1)


        self.dropout = Dropout(dropout_rate)

        self.age_avg_output = Dense(1, activation='relu', name='apparent_age_avg')
        self.age_std_output = Dense(1, activation='relu', name='apparent_age_std')

    def _get_dense(self, units):
        if self.use_flipout:
            return tfp.layers.DenseFlipout(units, activation=None)
        elif self.use_dropconnect:
            return DropConnectDense(units, dropconnect_rate=self.dropconnect_rate, activation=None)
        else:
            return Dense(units, activation=None)

    def call(self, inputs, training=None):
        x = self.base_model(inputs, training=training)
        x = self.global_avg_pool(x)

        x = self.dense1(x)
        x = self.batch_norm1(x)
        x = self.relu1(x)

        x = self.dense2(x)
        x = self.batch_norm2(x)
        x = self.relu2(x)

        x = self.dropout(x, training=training)

        return {
            "apparent_age_avg": self.age_avg_output(x),
            "apparent_age_std": self.age_std_output(x)
        }

    def train(self, train_data, valid_data, epochs=20, train_steps=1000, valid_steps=200):
        self.compile(
            optimizer='adam',
            loss={"apparent_age_avg": 'mse', "apparent_age_std": 'mse'}
        )

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            min_delta=0.001,
            restore_best_weights=True,
            verbose=1
        )

        history = self.fit(
            train_data,
            validation_data=valid_data,
            epochs=epochs,
            steps_per_epoch=train_steps,
            validation_steps=valid_steps,
            callbacks=[early_stopping]
        )
        return history

    def fine_tune(self, train_data, valid_data, epochs=5, train_steps=1000, valid_steps=200):
        for layer in self.base_model.layers[-100:]:
            layer.trainable = True

        self.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss={"apparent_age_avg": 'mse', "apparent_age_std": 'mse'}
        )

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            min_delta=0.001,
            restore_best_weights=True,
            verbose=1
        )

        history = self.fit(
            train_data,
            validation_data=valid_data,
            epochs=epochs,
            steps_per_epoch=train_steps,
            validation_steps=valid_steps,
            callbacks=[early_stopping]
        )
        return history

    def get_config(self):
        config = super(AgeEstimationModel, self).get_config()
        config.update({
            "input_shape": self.base_model.input_shape[1:],
            "dropout_rate": self.dropout.rate,
            "use_flipout": self.use_flipout,
            "use_dropconnect": self.use_dropconnect,
            "dropconnect_rate": self.dropconnect_rate
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def predict_age(self, img_path):
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        img_array = tf.expand_dims(img_array, axis=0)
        prediction = self.predict(img_array)
        return {
            "apparent_age_avg": prediction["apparent_age_avg"].numpy()[0][0],
            "apparent_age_std": prediction["apparent_age_std"].numpy()[0][0]
        }

class EnsembleAgeEstimator:
    def __init__(self, num_models=5, **model_kwargs):
        self.models = [AgeEstimationModel(**model_kwargs) for _ in range(num_models)]

    def compile_all(self):
        for model in self.models:
            model.compile(
                optimizer='adam',
                loss={"apparent_age_avg": 'mse', "apparent_age_std": 'mse'}
            )

    def fit_all(self, train_data, valid_data, epochs=20, train_steps=1000, valid_steps=200):
        histories = []
        for i, model in enumerate(self.models):
            print(f"Training model {i+1}/{len(self.models)}")
            histories.append(model.train(train_data, valid_data, epochs, train_steps, valid_steps))
        return histories

    def predict(self, img_path):
        preds_avg = []
        preds_std = []

        for model in self.models:
            pred = model.predict_age(img_path)
            preds_avg.append(pred["apparent_age_avg"])
            preds_std.append(pred["apparent_age_std"])

        mean_avg = tf.reduce_mean(preds_avg)
        mean_std = tf.reduce_mean(preds_std)

        return {
            "mean_apparent_age": mean_avg.numpy(),
            "mean_uncertainty": mean_std.numpy()
        }