from __future__ import annotations

import sys

import torch
import torch.nn
from torch import nn

from qualia_core.learningmodel.pytorch.layers.QuantizedLayer import QuantizedLayer

from .layers.quantized_layers import QuantizedBatchNorm
from .Quantizer import QuantizationConfig, Quantizer, update_params

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class QuantizedConv1d(nn.Conv1d, QuantizedLayer):
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
        self.call_super_init = True # Support multiple inheritance from nn.Module
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
            q_w = self.quantizer_w(self.weight, bias_tensor=self.bias)
            y = torch.nn.functional.conv1d(q_input,
                                           q_w,
                                           stride=self.stride,
                                           padding=self.padding,
                                           dilation = self.dilation,
                                           groups=self.groups)
        else :
            # Quantize bias and weights
            if hasattr(self, 'quantizer_bias'):
                #...with the different quantization schemes
                q_w = self.quantizer_w(self.weight)
                q_b = self.quantizer_bias(self.bias)
            else :
                #...with the same quantization schemes
                q_w, q_b = self.quantizer_w(self.weight, bias_tensor=self.bias)
            y = torch.nn.functional.conv1d(q_input,
                                           q_w,
                                           bias=q_b,
                                           stride=self.stride,
                                           padding=self.padding,
                                           dilation=self.dilation,
                                           groups=self.groups)

        if self.activation:
            y = self.activation(y)

        return self.quantizer_act(y)


class QuantizedMaxPool1d(nn.MaxPool1d, QuantizedLayer):
    def __init__(self,  # noqa: PLR0913
                 kernel_size: int,
                 quant_params: QuantizationConfig,
                 stride: int | None = None,
                 padding: int = 0,
                 activation: nn.Module | None = None) -> None:
        self.call_super_init = True # Support multiple inheritance from nn.Module
        super().__init__(kernel_size, stride=stride, padding=padding)
        self.activation = activation
        quant_params_input = update_params(tensor_type= 'input', quant_params=quant_params)
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


class QuantizedAdaptiveAvgPool1d(torch.nn.AdaptiveAvgPool1d, QuantizedLayer):
    def __init__(self,
                 output_size: int,
                 quant_params: QuantizationConfig,
                 activation: nn.Module | None = None) -> None:
        self.call_super_init = True # Support multiple inheritance from nn.Module
        super().__init__(output_size)
        self.activation = activation
        quant_params_input = update_params(tensor_type='input', quant_params=quant_params)
        quant_params_act = update_params(tensor_type='act', quant_params=quant_params)
        self.quantizer_input = Quantizer(**quant_params_input)
        self.quantizer_act = Quantizer(**quant_params_act)

    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        q_input = self.quantizer_input(input)

        y = super().forward(q_input)

        if self.activation:
            y = self.activation(y)

        return self.quantizer_act(y)


class QuantizedBatchNorm1d(QuantizedBatchNorm):
    ...


class QuantizedAvgPool1d(torch.nn.AvgPool1d, QuantizedLayer):
    def __init__(self,
                 kernel_size: int | tuple[int],
                 quant_params: QuantizationConfig,
                 stride: int | tuple[int] | None = None,
                 padding: int | tuple[int] = 0,
                 activation: nn.Module | None = None) -> None:
        self.call_super_init = True # Support multiple inheritance from nn.Module
        super().__init__(kernel_size, stride=stride, padding=padding)
        self.activation = activation
        quant_params_input = update_params(tensor_type='input', quant_params=quant_params)
        quant_params_act = update_params(tensor_type='act', quant_params=quant_params)
        self.quantizer_input = Quantizer(**quant_params_input)
        self.quantizer_act = Quantizer(**quant_params_act)

    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        q_input = self.quantizer_input(input)

        y = super().forward(q_input)

        if self.activation:
            y = self.activation(y)

        return self.quantizer_act(y)
