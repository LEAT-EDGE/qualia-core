import numpy as np

class FeatureSample:
    def __init__(self, features: 'List'=None):
        self.features = features if features is not None else []

    def get_raw_array(self):
        return np.reshape(self.features, (-1, 1))
