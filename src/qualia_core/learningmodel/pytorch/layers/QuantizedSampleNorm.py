from __future__ import annotations

import sys

from qualia_core.learningmodel.pytorch.Quantizer import Quantizer, update_params
from qualia_core.typing import TYPE_CHECKING, QuantizationConfigDict

from .QuantizedLayer import QuantizedLayer, QuantizerActProtocol, QuantizerInputProtocol
from .SampleNorm import SampleNorm

if TYPE_CHECKING:
    import torch  # noqa: TCH002
    from torch import nn  # noqa: TCH002

    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class QuantizedSampleNorm(SampleNorm, QuantizerInputProtocol, QuantizerActProtocol, QuantizedLayer):
    def __init__(self, norm: str, quant_params: QuantizationConfigDict) -> None:
        super().__init__(norm=norm)
        quant_params_input = update_params(tensor_type='input', quant_params=quant_params)
        quant_params_act = update_params(tensor_type='act', quant_params=quant_params)
        self.quantizer_input = Quantizer(**quant_params_input)
        self.quantizer_act = Quantizer(**quant_params_act)

    @classmethod
    @override
    def from_module(cls, module: nn.Module, quant_params: QuantizationConfigDict) -> Self:
        return cls(norm=module.norm, quant_params=quant_params)

    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        q_input = self.quantizer_input(input)

        y = super().forward(q_input)

        return self.quantizer_act(y)
