from torch.nn import AdaptiveAvgPool1d as AdaptiveAvgPool
from torch.nn import AvgPool1d as AvgPool
from torch.nn import BatchNorm1d as BatchNorm
from torch.nn import Conv1d as Conv
from torch.nn import MaxPool1d as MaxPool

from .GlobalSumPool1d import GlobalSumPool1d as GlobalSumPool
from .quantized_layers1d import QuantizedAdaptiveAvgPool1d as QuantizedAdaptiveAvgPool
from .quantized_layers1d import QuantizedAvgPool1d as QuantizedAvgPool
from .quantized_layers1d import QuantizedBatchNorm1d as QuantizedBatchNorm
from .quantized_layers1d import QuantizedConv1d as QuantizedConv
from .quantized_layers1d import QuantizedMaxPool1d as QuantizedMaxPool
from .QuantizedGlobalSumPool1d import QuantizedGlobalSumPool1d as QuantizedGlobalSumPool

__all__ = [
           'AdaptiveAvgPool',
           'AvgPool',
           'BatchNorm',
           'Conv',
           'GlobalSumPool',
           'MaxPool',
           'QuantizedAdaptiveAvgPool',
           'QuantizedAvgPool',
           'QuantizedBatchNorm',
           'QuantizedConv',
           'QuantizedGlobalSumPool',
           'QuantizedMaxPool',
]
