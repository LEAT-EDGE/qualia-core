import importlib.util
import logging

logger = logging.getLogger(__name__)

__all__: list[str] = []

if importlib.util.find_spec('tensorflow') is None:
    logger.warning('Warning: cannot find TensorFlow, Keras framework will be unavailable')
elif importlib.util.find_spec('tensorflow.keras') is None:
    logger.warning('Warning: cannot find TensorFlow.Keras, Keras framework will be unavailable')
else:
    from .Keras import Keras
    __all__ += ['Keras']

if importlib.util.find_spec('torch') is None:
    logger.warning('Warning: cannot find PyTorch, PyTorch framework will be unavailable')
elif importlib.util.find_spec('lightning.pytorch') is None:
    logger.warning('Warning: cannot find PyTorch Lightning, PyTorch framework will be unavailable')
else:
    from .PyTorch import PyTorch
    __all__ += ['PyTorch']
