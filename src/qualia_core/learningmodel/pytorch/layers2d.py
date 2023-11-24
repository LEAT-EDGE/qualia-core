from torch.nn import Conv2d as Conv
from torch.nn import BatchNorm2d as BatchNorm
from torch.nn import AvgPool2d as AvgPool
from torch.nn import AdaptiveAvgPool2d as AdaptiveAvgPool
from torch.nn import MaxPool2d as MaxPool
from .quantized_layers2d import QuantizedBatchNorm2d as QuantizedBatchNorm
from .quantized_layers2d import QuantizedConv2d as QuantizedConv
from .quantized_layers2d import QuantizedMaxPool2d as QuantizedMaxPool
from .quantized_layers2d import QuantizedAvgPool2d as QuantizedAvgPool
from .quantized_layers2d import QuantizedAdaptiveAvgPool2d as QuantizedAdaptiveAvgPool
from .layers import GlobalSumPool2d as GlobalSumPool
from .layers import QuantizedGlobalSumPool2d as QuantizedGlobalSumPool

__all__ = ['Conv',
           'BatchNorm',
           'AvgPool',
           'AdaptiveAvgPool',
           'MaxPool',
           'QuantizedBatchNorm',
           'QuantizedConv',
           'QuantizedMaxPool',
           'QuantizedAvgPool',
           'QuantizedAdaptiveAvgPool',
           'GlobalSumPool',
           'QuantizedGlobalSumPool']
