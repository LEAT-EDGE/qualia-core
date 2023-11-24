import importlib.util
import sys

__all__ = []

if importlib.util.find_spec("tensorflow") is None:
    print('Warning: cannot find TensorFlow, Keras models will be unavailable', file=sys.stderr)
elif importlib.util.find_spec("tensorflow.keras") is None:
    print('Warning: cannot find TensorFlow.Keras, Keras models will be unavailable', file=sys.stderr)
else:
    from . import keras
    __all__ += ['keras']

if importlib.util.find_spec("torch") is None:
    print('Warning: cannot find PyTorch, PyTorch models will be unavailable', file=sys.stderr)
elif importlib.util.find_spec("pytorch_lightning") is None:
    print('Warning: cannot find PyTorch Lightning, PyTorch models will be unavailable', file=sys.stderr)
else:
    from . import pytorch
    __all__ += ['pytorch']

