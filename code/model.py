import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import regularizers # type: ignore
from tensorflow.keras.applications import DenseNet121 # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization, LeakyReLU # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

class DropConnectDense(Dense):
    def __init__(self, units, dropconnect_rate=0.0, **kwargs):
        super(DropConnectDense, self).__init__(units, **kwargs)
        self.dropconnect_rate = dropconnect_rate

    def call(self, inputs, training=False):
        kernel = self.kernel
        if training and self.dropconnect_rate > 0:
            # Apply DropConnect to kernel
            random_tensor = tf.nn.dropout(tf.ones_like(kernel), rate=self.dropconnect_rate)
            kernel = tf.multiply(kernel, random_tensor)
        output = tf.matmul(inputs, kernel)
        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output


tfpl = tfp.layers


@tf.keras.utils.register_keras_serializable()
class AgeEstimationModel(tf.keras.Model):
    def __init__(self, input_shape=(224, 224, 3), dropout_rate=0.5, use_flipout=False, use_dropconnect=False, dropconnect_rate=0.3, **kwargs):
        super(AgeEstimationModel, self).__init__(**kwargs)
        self.use_flipout = use_flipout
        self.use_dropconnect = use_dropconnect
        self.dropout_rate = dropout_rate
        self.dropconnect_rate = dropconnect_rate

        self.base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
        self.base_model.trainable = False

        self.global_avg_pool = GlobalAveragePooling2D()

        self.dense1 = self.get_dense_layer(512)
        self.batch_norm1 = BatchNormalization()
        self.relu1 = LeakyReLU(alpha=0.1)

        self.dense2 = self.get_dense_layer(256)
        self.batch_norm2 = BatchNormalization()
        self.relu2 = LeakyReLU(alpha=0.1)

        self.age_avg_head = Dense(128, activation='relu')
        self.age_std_head = Dense(128, activation='relu')

        self.dropout = Dropout(dropout_rate)

        self.age_avg_output = Dense(1, activation='relu', name='apparent_age_avg')
        self.age_std_output = Dense(1, activation='softplus', name='apparent_age_std')


    def get_dense_layer(self, units):
        if self.use_flipout:
            return tfpl.DenseFlipout(units, activation=None)
        elif self.use_dropconnect:
            return DropConnectDense(units, dropconnect_rate=self.dropconnect_rate, activation=None)
        else:
            kernel_regularizer = regularizers.l2(1e-4)
            return Dense(units, activation=None, kernel_regularizer=kernel_regularizer)

    def call(self, inputs):
        x = self.base_model(inputs, training=True)
        x = self.global_avg_pool(x)

        x = self.dense1(x)
        x = self.batch_norm1(x)
        x = self.relu1(x)

        x = self.dense2(x)
        x = self.batch_norm2(x)
        x = self.relu2(x)

        x = self.dropout(x)

        avg_branch = self.age_avg_head(x)
        std_branch = self.age_std_head(x)

        return {
            "apparent_age_avg": self.age_avg_output(avg_branch),
            "apparent_age_std": self.age_std_output(std_branch)
        }

    def get_config(self):
        config = super(AgeEstimationModel, self).get_config()
        config.update({
            "input_shape": self.base_model.input_shape[1:],
            "dropout_rate": self.dropout_rate,
            "use_flipout": self.use_flipout,
            "use_dropconnect": self.use_dropconnect,
            "dropconnect_rate": self.dropconnect_rate
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(
            input_shape=config['input_shape'],
            dropout_rate=config['dropout_rate'],
            use_flipout=config.get('use_flipout', False),
            use_dropconnect=config.get('use_dropconnect', False),
            dropconnect_rate=config.get('dropconnect_rate', 0.3),
            name=config.get('name', None)
        )

    # Two-phase training: Phase 1 - Single task (only average)
    def train_single_task(self, train_data, valid_data, epochs=10, train_steps=1000, valid_steps=200):
            
        if self.use_flipout:
            new_optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
            new_patience = 10
        else:
            new_optimizer = 'adam'
            new_patience = 5
        self.compile(
            optimizer=new_optimizer,
            loss={
                "apparent_age_avg": 'mse',
                "apparent_age_std": lambda y_true, y_pred: tf.reduce_mean(y_pred * 0)
            },
            metrics={"apparent_age_avg": tf.keras.metrics.MeanAbsoluteError()}
        )

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            mode='min',
            patience=new_patience,
            restore_best_weights=True,
            verbose=1
        )

        history = self.fit(
            train_data,
            validation_data=valid_data,
            epochs=epochs,
            steps_per_epoch=train_steps,
            validation_steps=valid_steps,
            callbacks=[early_stopping],
            verbose=1
        )
        return history

    # Phase 2: Multitask loss (avg + std)
    def train_multitask(self, train_data, valid_data, epochs=10, train_steps=1000, valid_steps=200):
        layers_to_unfreeze = 100
        for layer in self.base_model.layers[-layers_to_unfreeze:]:
            layer.trainable = True

        new_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4 if not self.use_flipout else 5e-5)

        self.compile(
            optimizer=new_optimizer,
            loss={
                "apparent_age_avg": 'mse',
                "apparent_age_std": "mse"
            },
            metrics={
                "apparent_age_avg": [tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanSquaredError()],
                "apparent_age_std": [tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanSquaredError()]
            }
        )

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=7 if not self.use_flipout else 12,
            restore_best_weights=True,
            verbose=1
        )

        history = self.fit(
            train_data,
            validation_data=valid_data,
            epochs=epochs,
            steps_per_epoch=train_steps,
            validation_steps=valid_steps,
            callbacks=[early_stopping],
            verbose=1
        )
        return history

    def fine_tune(self, train_data, valid_data, epochs=5, train_steps=1000, valid_steps=200):
        # Unfreeze the last N layers of the DenseNet base model for fine-tuning
        layers_to_unfreeze = 200
        for layer in self.base_model.layers[-layers_to_unfreeze:]:
            layer.trainable = True

        # Re-compile with a smaller learning rate for fine-tuning
        self.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss={
                "apparent_age_avg": 'mse',
                "apparent_age_std": 'mse'
            },
            metrics={
                "apparent_age_avg": [tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanSquaredError()],
                "apparent_age_std": [tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanSquaredError()]
            }
        )

        # Train the model with fine-tuning
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            min_delta=0.001,
            restore_best_weights=True,
            verbose=1
        )

        fine_tune_history = self.fit(
            train_data,
            validation_data=valid_data,
            epochs=epochs,
            steps_per_epoch=train_steps,
            validation_steps=valid_steps,
            callbacks=[early_stopping],
            verbose=1
        )
        return fine_tune_history
