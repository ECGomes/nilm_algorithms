from .base_network import BaseNetwork
from .pb_base import PB_NILM
from .aux_functions import *

import tensorflow as tf


class Seq2Point(BaseNetwork):

    def __init__(self, x, y, window_size,
                 model_name, model_dir, dropout_rate=0.5, use_callbacks=False, stop_patience=50,
                 batch_size=128, n_epochs=100, validation_split=0.15):

        super().__init__(x, y, window_size, model_name, model_dir,
                         use_callbacks=use_callbacks, stop_patience=stop_patience,
                         batch_size=batch_size, n_epochs=n_epochs, validation_split=validation_split)

        # Set network specifics
        self.dropout_rate = dropout_rate

        # Normalization variable initialization
        self.standardized_x, self.x_mean, self.x_std = None, None, None
        self.standardized_y, self.y_mean, self.y_std = None, None, None

    def standardize_values(self, array):
        """
        Return the standardized inputs, as well as the mean and std
        """

        mean = array.mean()
        std = array.std()

        new_array = (array - mean) / std

        return new_array, mean, std

    def destandardize(self, array, mean, std):
        """
        Returns array multiplied by max value
        """
        return (array + mean) * std

    def preprocessing(self):
        """
        Normalizes by maximum value, then builds the sliding windows
        """

        self.standardized_x, self.x_mean, self.x_std = self.standardize_values(self.x)
        self.standardized_y, self.y_mean, self.y_std = self.standardize_values(self.y)

        preprocessed_x = split_sequence(self.standardized_x, self.window_size)
        preprocessed_x = preprocessed_x.reshape(preprocessed_x.shape[0],
                                                preprocessed_x.shape[1],
                                                1)
        preprocessed_x = tf.convert_to_tensor(preprocessed_x, dtype=tf.float32)
        preprocessed_y = tf.convert_to_tensor(self.standardized_y[self.window_size:], dtype=tf.float32)

        return preprocessed_x, preprocessed_y

    def network_architecture(self):

        inputs = tf.keras.layers.Input(shape=(self.window_size, 1))

        net = tf.keras.layers.Conv1D(filters=30, kernel_size=10, activation='relu', strides=1)(inputs)
        net = tf.keras.layers.Conv1D(filters=30, kernel_size=8, activation='relu', strides=1)(net)
        net = tf.keras.layers.Conv1D(filters=40, kernel_size=6, activation='relu', strides=1)(net)
        net = tf.keras.layers.Conv1D(filters=50, kernel_size=5, activation='relu', strides=1)(net)

        net = tf.keras.layers.Dropout(0.2)(net)

        net = tf.keras.layers.Conv1D(filters=50, kernel_size=5, activation='relu', strides=1)(net)

        net = tf.keras.layers.Dropout(0.2)(net)

        net = tf.keras.layers.Flatten()(net)
        net = tf.keras.layers.Dense(units=1024, activation='relu')(net)
        net = tf.keras.layers.Dropout(0.2)(net)

        # Changed from the original linear activation (so we don't run into negative consumption)
        output = tf.keras.layers.Dense(units=1, activation='relu')(net)

        model = tf.keras.Model(inputs, output)
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(patience=self.stop_patience, restore_best_weights=True)
        callbacks = [early_stopping]

        return model, callbacks

    def fit(self):

        # Get the data preprocessed
        processed_x, processed_y = self.preprocessing()

        # Get the network object
        self.model, callbacks = self.network_architecture()

        if self.use_callbacks:
            self.model.fit(processed_x, processed_y,
                           epochs=self.n_epochs,
                           batch_size=self.batch_size,
                           validation_split=self.validation_split,
                           callbacks=callbacks,
                           verbose=1)
        else:
            self.model.fit(processed_x, processed_y,
                           epochs=self.n_epochs,
                           batch_size=self.batch_size,
                           validation_split=self.validation_split,
                           verbose=1)
