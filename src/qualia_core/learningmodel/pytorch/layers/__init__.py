from . import quantized_layers
from .Add import Add
from .CustomBatchNorm1d import CustomBatchNorm1d
from .CustomBatchNorm2d import CustomBatchNorm2d
from .GlobalSumPool1d import GlobalSumPool1d
from .GlobalSumPool2d import GlobalSumPool2d
from .QuantizedAdd import QuantizedAdd
from .QuantizedGlobalSumPool1d import QuantizedGlobalSumPool1d
from .QuantizedGlobalSumPool2d import QuantizedGlobalSumPool2d
from .QuantizedSampleNorm import QuantizedSampleNorm
from .SampleNorm import SampleNorm

__all__ = [
        'Add',
        'CustomBatchNorm1d',
        'CustomBatchNorm2d',
        'GlobalSumPool1d',
        'GlobalSumPool2d',
        'QuantizedAdd',
        'QuantizedGlobalSumPool1d',
        'QuantizedGlobalSumPool2d',
        'QuantizedSampleNorm',
        'SampleNorm',
        ]

layers = (
        Add,
        CustomBatchNorm1d,
        CustomBatchNorm2d,
        QuantizedAdd,
        GlobalSumPool1d,
        GlobalSumPool2d,
        QuantizedGlobalSumPool1d,
        QuantizedGlobalSumPool2d,
        QuantizedSampleNorm,
        SampleNorm,
        *quantized_layers.quantized_layers,
        )
