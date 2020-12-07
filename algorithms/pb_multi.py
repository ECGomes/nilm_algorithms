from .base_network import BaseNetwork
from .pb_base import PB_NILM
from .aux_functions import *

import tensorflow as tf


class PB_Multi(PB_NILM):

    def preprocessing(self):
        preprocessed_x1 = split_sequence(self.x[:, 0], self.window_size)
        preprocessed_x1 = preprocessed_x1.reshape(preprocessed_x1.shape[0],
                                                  preprocessed_x1.shape[1],
                                                  1)
        preprocessed_x1 = tf.convert_to_tensor(preprocessed_x1, dtype=tf.float32)

        preprocessed_x2 = split_sequence(self.x[:, 1], self.window_size)
        preprocessed_x2 = preprocessed_x2.reshape(preprocessed_x2.shape[0],
                                                  preprocessed_x2.shape[1],
                                                  1)
        preprocessed_x2 = tf.convert_to_tensor(preprocessed_x2, dtype=tf.float32)

        preprocessed_x3 = split_sequence(self.x[:, 2], self.window_size)
        preprocessed_x3 = preprocessed_x3.reshape(preprocessed_x3.shape[0],
                                                  preprocessed_x3.shape[1],
                                                  1)
        preprocessed_x3 = tf.convert_to_tensor(preprocessed_x3, dtype=tf.float32)

        preprocessed_y1 = tf.convert_to_tensor(self.y[:, 0][self.window_size:], dtype=tf.float32)
        preprocessed_y2 = tf.convert_to_tensor(self.y[:, 1][self.window_size:], dtype=tf.float32)
        preprocessed_y3 = tf.convert_to_tensor(self.y[:, 2][self.window_size:], dtype=tf.float32)

        return [preprocessed_x1, preprocessed_x2, preprocessed_x3], [preprocessed_y1, preprocessed_y2, preprocessed_y3]

    def network_architecture(self):

        def pb_loss(y_true, y_pred):
            tau = self.pb_value
            err = y_true - y_pred

            if tf.__version__ < '2.0.0':
                return tf._api.v1.keras.backend.mean(tf._api.v1.keras.backend.maximum(tau * err, (tau - 1) * err),
                                                     axis=-1)
            else:
                return tf.keras.backend.mean(tf.keras.backend.maximum(tau * err, (tau - 1) * err), axis=-1)

        inputs = tf.keras.layers.Input(shape=(self.window_size, 1))

        # Define the first branch of the network
        branch1 = tf.keras.layers.Conv1D(128, kernel_size=5)(inputs)
        branch1 = tf.keras.layers.BatchNormalization()(branch1)
        branch1 = tf.keras.layers.Activation('relu')(branch1)

        if self.use_maxpool:
            branch1 = tf.keras.layers.MaxPooling1D()(branch1)

        branch1 = tf.keras.layers.GRU(256)(branch1)
        branch1 = tf.keras.layers.BatchNormalization()(branch1)
        branch1 = tf.keras.layers.Activation('relu')(branch1)

        if self.use_dropout:
            branch1 = tf.keras.layers.Dropout(rate=self.dropout_rate)(branch1)

        branch1 = tf.keras.layers.Dense(1, activation='relu', name='Appliance')(branch1)

        # Define the first branch of the network
        branch2 = tf.keras.layers.Conv1D(128, kernel_size=5)(inputs)
        branch2 = tf.keras.layers.BatchNormalization()(branch2)
        branch2 = tf.keras.layers.Activation('relu')(branch2)

        if self.use_maxpool:
            branch2 = tf.keras.layers.MaxPooling1D()(branch2)

        branch2 = tf.keras.layers.GRU(256)(branch2)
        branch2 = tf.keras.layers.BatchNormalization()(branch2)
        branch2 = tf.keras.layers.Activation('relu')(branch2)

        if self.use_dropout:
            branch2 = tf.keras.layers.Dropout(rate=self.dropout_rate)(branch2)

        branch2 = tf.keras.layers.Dense(1, activation='relu', name='Diff')(branch2)

        branch3 = tf.keras.layers.Add(name='Total')([branch1, branch2])

        early_stopping = tf.keras.callbacks.EarlyStopping(patience=self.stop_patience, restore_best_weights=True)

        model = tf.keras.Model(inputs=inputs, outputs=[branch1, branch2, branch3])
        model.compile(optimizer='adam', loss=pb_loss)

        return model, [early_stopping]
