from tensorflow.keras.optimizers import Optimizer
import tensorflow as tf

class DummyOptimizer(Optimizer):
    def __init__(self, name='DSOMOptimizer', **kwargs):
        super().__init__(name, **kwargs)

    def _create_slots(self, var_list):
        pass

    def minimize(self):
        pass

    def apply_gradients(self, grads_and_vars, name=None):
        pass

    def get_config(self):
        return super().get_config()
