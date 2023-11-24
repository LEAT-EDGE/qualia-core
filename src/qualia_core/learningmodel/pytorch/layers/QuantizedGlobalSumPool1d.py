from __future__ import annotations

import sys
from .GlobalSumPool1d import GlobalSumPool1d

import torch
from qualia_core.learningmodel.pytorch.Quantizer import Quantizer, update_params
from qualia_core.typing import RecursiveConfigDict
from torch import nn

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class QuantizedGlobalSumPool1d(GlobalSumPool1d):
    def __init__(self, quant_params: RecursiveConfigDict | None = None, activation: nn.Module | None = None) -> None:
        super().__init__()
        self.activation = activation
        quant_params_input = update_params(tensor_type='input', quant_params=quant_params)
        quant_params_act = update_params(tensor_type='act', quant_params=quant_params)
        self.quantizer_input = Quantizer(**quant_params_input)
        self.quantizer_act = Quantizer(**quant_params_act)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_input = self.quantizer_input(x)

        y = super().forward(q_input)

        if self.activation:
            y = self.activation(y)

        q_output = self.quantizer_act(y)

        return q_output

    @property
    def input_q(self) -> int | None:
        return self.quantizer_input.fractional_bits

    @property
    def activation_q(self) -> int | None:
        return self.quantizer_act.fractional_bits

    @property
    def weights_q(self) -> int | None:
        return None
