from .base_network import BaseNetwork
from .aux_functions import *

import tensorflow as tf

class PB_NILM(BaseNetwork):

    def __init__(self, x, y, window_size,
                 model_name, model_dir,
                 use_callbacks=True, stop_patience=50,
                 pb_value=0.50, batch_size=128, n_epochs=100, validation_split=0.15,
                 use_dropout=True, use_maxpool=False,
                 dropout_rate=0.15):
        super(PB_NILM, self).__init__(x, y, window_size, model_name, model_dir,
                                        use_callbacks=use_callbacks, stop_patience=stop_patience,
                                        batch_size=batch_size, n_epochs=n_epochs,
                                        validation_split=validation_split)
        self.use_dropout = use_dropout
        self.use_maxpool = use_maxpool
        self.dropout_rate = dropout_rate
        self.pb_value = pb_value

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