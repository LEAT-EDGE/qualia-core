from __future__ import annotations

import sys

import torch
import torch.nn
from torch import nn

from qualia_core.learningmodel.pytorch.Quantizer import QuantizationConfig, Quantizer, update_params
from qualia_core.typing import TYPE_CHECKING, QuantizationConfigDict

from .CustomBatchNorm import CustomBatchNorm
from .QuantizedLayer import QuantizedLayer, QuantizerActProtocol, QuantizerInputProtocol, QuantizerWProtocol

if TYPE_CHECKING:
    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override


class QuantizedLinear(nn.Linear, QuantizerInputProtocol, QuantizerActProtocol, QuantizerWProtocol, QuantizedLayer):
    bias: torch.nn.Parameter | None

    def __init__(self,  # noqa: PLR0913
                 in_features: int,
                 out_features: int,
                 quant_params: QuantizationConfig,
                 bias: bool = True,  # noqa: FBT001, FBT002
                 activation: nn.Module | None = None) -> None:
        self.call_super_init = True # Support multiple inheritance from nn.Module
        super().__init__(in_features, out_features, bias=bias)
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        # Create the quantizer instance
        quant_params_input = update_params(tensor_type = 'input', quant_params = quant_params)
        quant_params_act = update_params(tensor_type = 'act', quant_params = quant_params)
        quant_params_w = update_params(tensor_type = 'w', quant_params = quant_params)
        self.quantizer_input = Quantizer(**quant_params_input)
        self.quantizer_act = Quantizer(**quant_params_act)
        self.quantizer_w = Quantizer(**quant_params_w)
        if 'bias' in quant_params:
            quant_params_bias = update_params(tensor_type='bias', quant_params=quant_params)
            self.quantizer_bias = Quantizer(**quant_params_bias)

    @classmethod
    @override
    def from_module(cls, module: nn.Module, quant_params: QuantizationConfigDict) -> Self:
        quantized_module = cls(in_features=module.in_features,
                               out_features=module.out_features,
                               quant_params=quant_params)
        with torch.no_grad():
            _ = quantized_module.weight.copy_(module.weight)
            if quantized_module.bias is not None:
                _ = quantized_module.bias.copy_(module.bias)
        return quantized_module

    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        #if self.quan_train and self.quand_test:

        q_input = self.quantizer_input(input)
        if self.bias is None:
            # If no bias, quantize weights only
            q_w = self.quantizer_w(self.weight)
            y = torch.nn.functional.linear(q_input, q_w)
        else :
            # Quantize bias and weights
            if self.quantizer_bias is not None:
                #...with the different quantization schemes
                q_w = self.quantizer_w(self.weight)
                q_b = self.quantizer_bias(self.bias)
            else :
                #...with the same quantization schemes
                q_w, q_b = self.quantizer_w(self.weight, bias_tensor=self.bias)
            y = torch.nn.functional.linear(q_input, q_w, bias=q_b)

        if self.activation:
            y = self.activation(y)

        return self.quantizer_act(y)


class QuantizedReLU(torch.nn.ReLU, QuantizerInputProtocol, QuantizerActProtocol, QuantizedLayer):
    def __init__(self, quant_params: QuantizationConfig) -> None:
        self.call_super_init = True # Support multiple inheritance from nn.Module
        super().__init__()
        quant_params_input = update_params(tensor_type='input', quant_params=quant_params)
        quant_params_act = update_params(tensor_type='act', quant_params=quant_params)
        self.quantizer_input = Quantizer(**quant_params_input)
        self.quantizer_act = Quantizer(**quant_params_act)

    @classmethod
    @override
    def from_module(cls, module: nn.Module, quant_params: QuantizationConfigDict) -> Self:
        return cls(quant_params=quant_params)

    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        q_input = self.quantizer_input(input)

        y = super().forward(q_input)

        return self.quantizer_act(y)


class QuantizedIdentity(torch.nn.Identity, QuantizerInputProtocol, QuantizerActProtocol, QuantizedLayer):
    def __init__(self, quant_params: QuantizationConfig) -> None:
        self.call_super_init = True # Support multiple inheritance from nn.Module
        super().__init__()
        quant_params_act = update_params(tensor_type='act', quant_params=quant_params)
        self.quantizer_act = Quantizer(**quant_params_act)

        # Identity has same quantization on input and output
        # Only output quantizer used here but generation of activations_range.txt require input quantized info as well
        self.quantizer_input = self.quantizer_act

    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002

        y = super().forward(input)

        return self.quantizer_act(y)


class QuantizedBatchNorm(CustomBatchNorm, QuantizerInputProtocol, QuantizerActProtocol, QuantizerWProtocol, QuantizedLayer):
    def __init__(self,  # noqa: PLR0913
                 num_features: int,
                 quant_params: QuantizationConfig,
                 eps: float = 1e-05,
                 momentum: float = 0.1,
                 affine: bool = True,  # noqa: FBT001, FBT002
                 track_running_stats: bool = True,  # noqa: FBT002, FBT001
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None,
                 activation: torch.nn.Module | None = None) -> None:
        super().__init__(num_features=num_features,
                         eps=eps,
                         momentum=momentum,
                         affine=affine,
                         track_running_stats=track_running_stats,
                         use_fused_params=True,
                         device=device,
                         dtype=dtype)

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

    @classmethod
    @override
    def from_module(cls, module: nn.Module, quant_params: QuantizationConfigDict) -> Self:
        quantized_module = cls(num_features=module.num_features,
                               eps=module.eps,
                               momentum=module.momentum,
                               affine=module.affine,
                               track_running_stats=module.track_running_stats,
                               device=module.device,
                               dtype=module.dtype,
                               quant_params=quant_params)
        with torch.no_grad():
            if quantized_module.affine:
                _ = quantized_module.weight.copy_(module.weight)
                _ = quantized_module.bias.copy_(module.bias)
            if quantized_module.track_running_stats:
                _ = quantized_module.running_var.copy_(module.running_var)
                _ = quantized_module.running_mean.copy_(module.running_mean)
                _ = quantized_module.num_batches_tracked.copy_(module.num_batches_tracked)
        return quantized_module

    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002

        input_shape = input.shape
        x = input.flatten(start_dim=2)
        q_input = self.quantizer_input(x)

        mean, var = self.compute_stats(q_input)

        if self.training and self.track_running_stats:
            self.update_running_stats(mean=mean, var=var)

        if self.training or not self.track_running_stats:
            alpha, beta = self.compute_fused_params(weight=self.weight,
                                                    bias=self.bias,
                                                    mean=mean,
                                                    var=var,
                                                    eps=self.eps)
        else:
            alpha, beta = self.compute_fused_params(weight=self.weight,
                                                    bias=self.bias,
                                                    mean=self.running_mean,
                                                    var=self.running_var,
                                                    eps=self.eps)

        # Quantize bias and weights
        if self.quantizer_bias is not None:
            #...with the different quantization schemes
            q_alpha = self.quantizer_w(alpha)
            q_beta = self.quantizer_bias(beta)
        else :
            #...with the same quantization schemes
            q_alpha, q_beta = self.quantizer_w(alpha, bias_tensor=beta)

        y = self.compute_batchnorm_with_fused_params(x=q_input, alpha=q_alpha, beta=q_beta)

        if self.activation is not None:
            y = self.activation(y)

        return self.quantizer_act(y).reshape(input_shape)


# Keep imports there to avoid circular imports

from qualia_core.learningmodel.pytorch.quantized_layers1d import (  # noqa: E402
    QuantizedAdaptiveAvgPool1d,
    QuantizedAvgPool1d,
    QuantizedBatchNorm1d,
    QuantizedConv1d,
    QuantizedMaxPool1d,
)
from qualia_core.learningmodel.pytorch.quantized_layers2d import (  # noqa: E402
    QuantizedAdaptiveAvgPool2d,
    QuantizedAvgPool2d,
    QuantizedBatchNorm2d,
    QuantizedConv2d,
    QuantizedMaxPool2d,
)

from .QuantizedAdd import QuantizedAdd  # noqa: E402
from .QuantizedGlobalSumPool1d import QuantizedGlobalSumPool1d  # noqa: E402
from .QuantizedGlobalSumPool2d import QuantizedGlobalSumPool2d  # noqa: E402

quantized_layers = (
        QuantizedBatchNorm1d,
        QuantizedBatchNorm2d,
        QuantizedConv1d,
        QuantizedConv2d,
        QuantizedMaxPool1d,
        QuantizedMaxPool2d,
        QuantizedAvgPool1d,
        QuantizedAvgPool2d,
        QuantizedAdaptiveAvgPool1d,
        QuantizedAdaptiveAvgPool2d,
        QuantizedLinear,
        QuantizedAdd,
        QuantizedReLU,
        QuantizedIdentity,
        QuantizedGlobalSumPool1d,
        QuantizedGlobalSumPool2d,
        )
