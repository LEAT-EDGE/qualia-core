from __future__ import annotations

import sys
from typing import Optional, Union

import torch
import torch.nn
from torch import nn

from qualia_core.typing import TYPE_CHECKING

from .layers.quantized_layers import QuantizedBatchNorm
from .Quantizer import QuantizationConfig, Quantizer, update_params

if TYPE_CHECKING:
    from qualia_core.typing import RecursiveConfigDict

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class QuantizedConv2d(nn.Conv2d):
    def __init__(self,  # noqa: PLR0913
                 in_channels: int,
                 out_channels: int,
                 quant_params: QuantizationConfig,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True,  # noqa: FBT001, FBT002
                 activation: nn.Module | None = None) -> None:
        super().__init__(in_channels,
                         out_channels,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding,
                         dilation=dilation,
                         groups=groups,
                         bias=bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_width = kernel_size
        self.groups = groups
        self.activation = activation
        # Create the quantizer instance
        quant_params_input = update_params(tensor_type='input', quant_params=quant_params)
        quant_params_act = update_params(tensor_type='act', quant_params=quant_params)
        quant_params_w = update_params(tensor_type='w', quant_params=quant_params)
        self.quantizer_input = Quantizer(**quant_params_input)
        self.quantizer_act = Quantizer(**quant_params_act)
        self.quantizer_w = Quantizer(**quant_params_w)
        if 'bias' in quant_params :
            quant_params_bias = update_params(tensor_type='bias', quant_params=quant_params)
            self.quantizer_bias = Quantizer(**quant_params_bias)

    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        q_input = self.quantizer_input(input)

        if self.bias is None :
            # If no bias, quantize weights only
            q_w = self.quantizer_w(self.weight)
            y = torch.nn.functional.conv2d(q_input,
                                           q_w,
                                           stride=self.stride,
                                           padding=self.padding,
                                           dilation = self.dilation,
                                           groups=self.groups)
        else:
            # Quantize bias and weights
            if hasattr(self, 'quantizer_bias'):
                #...with the different quantization schemes
                q_w = self.quantizer_w(self.weight)
                q_b = self.quantizer_bias(self.bias)
            else :
                #...with the same quantization schemes
                q_w, q_b = self.quantizer_w(self.weight, bias_tensor=self.bias)
            y = torch.nn.functional.conv2d(q_input,
                                           q_w,
                                           bias=q_b,
                                           stride=self.stride,
                                           padding=self.padding,
                                           dilation=self.dilation,
                                           groups=self.groups)

        if self.activation:
            y = self.activation(y)

        return self.quantizer_act(y)

    @property
    def input_q(self) -> int | None:
        return self.quantizer_input.fractional_bits

    @property
    def activation_q(self) -> int | None:
        return self.quantizer_act.fractional_bits

    @property
    def weights_q(self) -> int | None:
        return self.quantizer_w.fractional_bits


class QuantizedMaxPool2d(nn.MaxPool2d):
    def __init__(self,  # noqa: PLR0913
                 kernel_size: int,
                 quant_params: QuantizationConfig,
                 stride: int | None = None,
                 padding: int =0,
                 activation: nn.Module | None = None) -> None:
        super().__init__(kernel_size, stride=stride, padding=padding)
        self.activation = activation
        quant_params_input = update_params(tensor_type='input', quant_params=quant_params)
        quant_params_act = update_params(tensor_type='act', quant_params=quant_params)
        self.quantizer_input = Quantizer(**quant_params_input)
        self.quantizer_act = Quantizer(**quant_params_act)

    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002

        q_input = self.quantizer_input(input)

        y: torch.Tensor = super().forward(q_input)

        if self.activation:
            y = self.activation(y)

        return self.quantizer_act(y)

    @property
    def input_q(self) -> int | None:
        return self.quantizer_input.fractional_bits

    @property
    def activation_q(self) -> int | None:
        return self.quantizer_act.fractional_bits

    @property
    def weights_q(self) -> int | None:
        return None


class QuantizedAdaptiveAvgPool2d(torch.nn.AdaptiveAvgPool2d):
    def __init__(self, output_size, quant_params = None, activation=None):
        super().__init__(output_size)
        self.activation = activation
        quant_params_input = update_params(tensor_type = "input", quant_params = quant_params)
        quant_params_act = update_params(tensor_type = "act", quant_params = quant_params)
        self.quantizer_input = Quantizer(**quant_params_input)
        self.quantizer_act = Quantizer(**quant_params_act)

 
    def forward(self, input):

        q_input = self.quantizer_input(input)
        
        y = super().forward(q_input)

        if self.activation:
            y = self.activation(y)

        q_output = self.quantizer_act(y)

        return q_output

    @property
    def input_q(self) -> Optional[int]:
        return self.quantizer_input.fractional_bits

    @property
    def activation_q(self) -> Optional[int]:
        return self.quantizer_act.fractional_bits

    @property
    def weights_q(self) -> Optional[int]:
        return None


class QuantizedBatchNorm2d(QuantizedBatchNorm):
    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        """Reshape to 1D-like input in order to use QuantizedBatchNorm forward as-is.

        BatchNorm works on channels with are second dim after batch so flatten the last dimensions to single dim.
        """
        input_shape = input.shape
        input = input.flatten(start_dim=2)

        y = super().forward(input)

        return y.reshape(input_shape)


class QuantizedAvgPool2d(torch.nn.AvgPool2d):
    def __init__(self,
                 kernel_size: Union[int, tuple[int]],
                 stride: Optional[Union[int, tuple[int]]] = None,
                 padding: Union[int, tuple[int]] = 0,
                 quant_params: Optional[RecursiveConfigDict]=None,
                 activation: Optional[torch.nn.Module]=None) -> None:
        super().__init__(kernel_size, stride=stride, padding=padding)
        self.activation = activation
        quant_params_input = update_params(tensor_type='input', quant_params=quant_params)
        quant_params_act = update_params(tensor_type='act', quant_params=quant_params)
        self.quantizer_input = Quantizer(**quant_params_input)
        self.quantizer_act = Quantizer(**quant_params_act)

    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        q_input = self.quantizer_input(input)

        y = super().forward(q_input)

        if self.activation:
            y = self.activation(y)

        q_output = self.quantizer_act(y)

        return q_output

    @property
    def input_q(self) -> Optional[int]:
        return self.quantizer_input.fractional_bits

    @property
    def activation_q(self) -> Optional[int]:
        return self.quantizer_act.fractional_bits

    @property
    def weights_q(self) -> Optional[int]:
        return None
