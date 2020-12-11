from .base_network import BaseNetwork
from .pb_base import PB_NILM
from .aux_functions import *

import tensorflow as tf


class WindowGRU(BaseNetwork):

    def __init__(self, x, y, window_size,
                 model_name, model_dir, dropout_rate=0.5, use_callbacks=False, stop_patience=50,
                 batch_size=128, n_epochs=100, validation_split=0.15):

        super().__init__(x, y, window_size, model_name, model_dir,
                         use_callbacks=use_callbacks, stop_patience=stop_patience,
                         batch_size=batch_size, n_epochs=n_epochs, validation_split=validation_split)

        # Set network specifics
        self.dropout_rate = dropout_rate

        # Normalization variable initialization
        self.max_val = self.get_max()

        self.normalized_x = None
        self.normalized_y = None

    def get_max(self):
        """
        Get maximum value between X and Y
        """
        max_x = self.x.max()
        max_y = self.y.max()

        return max(max_x, max_y)

    def normalize(self, array):
        """
        Returns array normalized (divided) by max value
        """
        return array/self.max_val

    def denormalize(self, array):
        """
        Returns array multiplied by max value
        """
        return array * self.max_val

    def preprocessing(self):
        """
        Normalizes by maximum value, then builds the sliding windows
        """

        self.normalized_x = self.normalize(self.x)
        self.normalized_y = self.normalize(self.y)

        preprocessed_x = split_sequence(self.normalized_x, self.window_size)
        preprocessed_x = preprocessed_x.reshape(preprocessed_x.shape[0],
                                                preprocessed_x.shape[1],
                                                1)
        preprocessed_x = tf.convert_to_tensor(preprocessed_x, dtype=tf.float32)
        preprocessed_y = tf.convert_to_tensor(self.normalized_y[self.window_size:], dtype=tf.float32)

        return preprocessed_x, preprocessed_y

    def denormalize(self, array):
        """
        De-normalizes the sequence
        """

        return array * self.max_val

    def network_architecture(self):

        inputs = tf.keras.layers.Input(shape=(self.window_size, 1))

        net = tf.keras.layers.Conv1D(filters=16, kernel_size=4, activation='relu',
                                     padding='same')(inputs)
        net = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=64, activation='relu',
                                                                return_sequences=True), merge_mode='concat')(net)
        net = tf.keras.layers.Dropout(self.dropout_rate)(net)
        net = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=128, activation='relu',
                                                                return_sequences=True), merge_mode='concat')(net)
        net = tf.keras.layers.Dropout(self.dropout_rate)(net)

        net = tf.keras.layers.Dense(units=128, activation='relu')(net)
        net = tf.keras.layers.Dropout(self.dropout_rate)(net)

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
