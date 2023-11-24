import importlib.util
import logging

from .Distribution import Distribution
from .Keras2TFLite import Keras2TFLite
from .PostProcessing import PostProcessing
from .QualiaCodeGen import QualiaCodeGen

__all__ = [
        'Distribution',
        'FuseBatchNorm',
        'Keras2TFLite',
        'PostProcessing',
        'QualiaCodeGen',
        ]

logger = logging.getLogger(__name__)

if importlib.util.find_spec('torch') is None:
    logger.warning('PyTorch is required for FuseBatchNorm, QuantizationAwareTraining, Torch2Keras')
else:
    from .FuseBatchNorm import FuseBatchNorm
    from .QuantizationAwareTraining import QuantizationAwareTraining

    __all__ += ['FuseBatchNorm',
                'QuantizationAwareTraining']

if importlib.util.find_spec('keras') is None:
    logger.warning('Keras is required for RemoveKerasSoftmax, Torch2Keras')
else:
    from .RemoveKerasSoftmax import RemoveKerasSoftmax

    __all__ += ['RemoveKerasSoftmax']

    # Warning message already printed
    if importlib.util.find_spec('torch') is not None:
        from .Torch2Keras import Torch2Keras

        __all__ += ['Torch2Keras']
