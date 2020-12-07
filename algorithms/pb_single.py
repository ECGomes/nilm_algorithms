from .base_network import BaseNetwork
from .pb_base import PB_NILM
from .aux_functions import *

import tensorflow as tf


class PB_Single(PB_NILM):

    def preprocessing(self):
        preprocessed_x = split_sequence(self.x, self.window_size)
        preprocessed_x = preprocessed_x.reshape(preprocessed_x.shape[0],
                                                preprocessed_x.shape[1],
                                                1)
        preprocessed_x = tf.convert_to_tensor(preprocessed_x, dtype=tf.float32)
        preprocessed_y = tf.convert_to_tensor(self.y[self.window_size:], dtype=tf.float32)

        return preprocessed_x, preprocessed_y


    def network_architecture(self):
        """
        Returns the network for the PB-NILM single branch version from the paper:
        https://ieeexplore.ieee.org/document/9025262
        """

        def pb_loss(y_true, y_pred):
            tau = self.pb_value
            err = y_true - y_pred

            if tf.__version__ < '2.0.0':
                return tf._api.v1.keras.backend.mean(tf._api.v1.keras.backend.maximum(tau * err, (tau - 1) * err),
                                                     axis=-1)
            else:
                return tf.keras.backend.mean(tf.keras.backend.maximum(tau * err, (tau - 1) * err), axis=-1)

        inputs = tf.keras.layers.Input(shape=(self.window_size, 1))

        # Define the network
        net = tf.keras.layers.Conv1D(128, kernel_size=5)(inputs)
        net = tf.keras.layers.BatchNormalization()(net)
        net = tf.keras.layers.Activation('relu')(net)

        if self.use_maxpool:
            net = tf.keras.layers.MaxPooling1D()(net)

        net = tf.keras.layers.GRU(256)(net)
        net = tf.keras.layers.BatchNormalization()(net)
        net = tf.keras.layers.Activation('relu')(net)

        if self.use_dropout:
            net = tf.keras.layers.Dropout(rate=self.dropout_rate)(net)

        net = tf.keras.layers.Dense(1, activation='relu', name='Appliance')(net)

        early_stopping = tf.keras.callbacks.EarlyStopping(patience=self.stop_patience, restore_best_weights=True)

        model = tf.keras.Model(inputs=inputs, outputs=net)
        model.compile(optimizer='adam', loss=pb_loss)

        return model, [early_stopping]