import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping

@tf.keras.utils.register_keras_serializable()
class AgeEstimationModel(tf.keras.Model):
    def __init__(self, input_shape=(224, 224, 3), dropout_rate=0.5, **kwargs):
        super(AgeEstimationModel, self).__init__(**kwargs)
        self.base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
        self.base_model.trainable = False  # Initially freeze all layers

        self.global_avg_pool = GlobalAveragePooling2D()
        
        # First dense layer
        self.dense1 = Dense(1024, activation=None)  # Larger number of neurons
        self.batch_norm1 = BatchNormalization()
        self.relu1 = LeakyReLU(alpha=0.1)  # Leaky ReLU for better training stability
        
        # Second dense layer
        self.dense2 = Dense(512, activation=None)
        self.batch_norm2 = BatchNormalization()
        self.relu2 = LeakyReLU(alpha=0.1)

        # Third dense layer
        self.dense3 = Dense(256, activation=None)
        self.batch_norm3 = BatchNormalization()
        self.relu3 = LeakyReLU(alpha=0.1)

        self.dropout = Dropout(dropout_rate)

        # Output layers for age prediction
        self.age_avg_output = Dense(1, activation='relu', name='apparent_age_avg')
        self.age_std_output = Dense(1, activation='relu', name='apparent_age_std') 

    def call(self, inputs):
        x = self.base_model(inputs, training=True)
        x = self.global_avg_pool(x)

        x = self.dense1(x)
        x = self.batch_norm1(x)
        x = self.relu1(x)

        x = self.dense2(x)
        x = self.batch_norm2(x)
        x = self.relu2(x)

        x = self.dense3(x)
        x = self.batch_norm3(x)
        x = self.relu3(x)

        x = self.dropout(x)

        return {
            "apparent_age_avg": self.age_avg_output(x),
            "apparent_age_std": self.age_std_output(x)
        }
    
    def train(self, train_data, valid_data, epochs=20, train_steps=1000, valid_steps=200):
        self.compile(
            optimizer='adam',
            loss={
                "apparent_age_avg": 'mse',
                "apparent_age_std": 'mse'
            }
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
        layers_to_unfreeze = 40
        for layer in self.base_model.layers[-layers_to_unfreeze:]:
            layer.trainable = True
        
        self.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss={
                "apparent_age_avg": 'mse',
                "apparent_age_std": 'mse'
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
            callbacks=[early_stopping]
        )
        return fine_tune_history

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

    def predict_age(self, img_path):
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        img_array = tf.expand_dims(img_array, axis=0)
        return self.predict(img_array)[0][0]
