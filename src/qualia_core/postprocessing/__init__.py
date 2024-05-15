import importlib.util
import logging

from .Distribution import Distribution
from .PostProcessing import PostProcessing
from .QualiaCodeGen import QualiaCodeGen

__all__ = [
        'Distribution',
        'PostProcessing',
        'QualiaCodeGen',
        ]

logger = logging.getLogger(__name__)

if importlib.util.find_spec('torch') is None:
    logger.warning('PyTorch is required for FuseBatchNorm, QuantizationAwareTraining, Torch2Keras')
else:
    from .FuseBatchNorm import FuseBatchNorm
    from .QuantizationAwareTraining import QuantizationAwareTraining
    from .QuantizationAwareTrainingFX import QuantizationAwareTrainingFX

    __all__ += ['FuseBatchNorm',
                'QuantizationAwareTraining',
                'QuantizationAwareTrainingFX']

if importlib.util.find_spec('keras') is None or importlib.util.find_spec('tensorflow') is None:
    logger.warning('TensorFlow and Keras are required for RemoveKerasSoftmax, Torch2Keras')
else:
    from .Keras2TFLite import Keras2TFLite
    from .RemoveKerasSoftmax import RemoveKerasSoftmax

    __all__ += ['Keras2TFLite', 'RemoveKerasSoftmax']

    # Warning message already printed
    if importlib.util.find_spec('torch') is not None:
        from .Torch2Keras import Torch2Keras

        __all__ += ['Torch2Keras']
