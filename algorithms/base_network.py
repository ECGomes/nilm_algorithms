import numpy as np
import pandas as pd
import os

class BaseNetwork(object):

    def __init__(self, x, y, window_size,
                 model_name, model_dir, use_callbacks=True, stop_patience=50,
                 batch_size=128, n_epochs=100, validation_split=0.15):

        # Data
        self.x = x
        self.y = y

        # Network Specifications
        self.window_size = window_size
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.validation_split = validation_split
        self.stop_patience = stop_patience

        # Model Specifics
        self.model = None
        self.model_name = model_name
        self.model_dir = model_dir
        self.use_callbacks = use_callbacks

    def preprocessing(self):
        raise NotImplementedError()

    def network_architecture(self):
        raise NotImplementedError()

    def fit(self):
        raise NotImplementedError()

    def save(self):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        model_path = os.path.join(self.model_dir, self.model_name + '.h5')
        self.model.save(model_path)
