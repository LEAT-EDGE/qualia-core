from __future__ import annotations

import sys
from dataclasses import dataclass

from qualia_core.learningmodel.pytorch.Quantizer import QuantizationConfig, Quantizer, update_params
from qualia_core.typing import TYPE_CHECKING

from .Add import Add
from .QuantizedLayer import QuantizedLayer, QuantizerActProtocol

if TYPE_CHECKING:
    import torch  # noqa: TCH002
    from torch import nn  # noqa: TCH002

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

@dataclass
class DummyInputQuantizer:
    global_max: None = None

class QuantizedAdd(Add, QuantizerActProtocol, QuantizedLayer):
    def __init__(self,
                 quant_params: QuantizationConfig,
                 activation: nn.Module | None = None) -> None:
        super().__init__()
        self.activation = activation
        quant_params_input = update_params(tensor_type='input', quant_params=quant_params)
        quant_params_act = update_params(tensor_type='act', quant_params=quant_params)
        self.quantizer_input_a = Quantizer(**quant_params_input)
        self.quantizer_input_b = Quantizer(**quant_params_input)
        self.quantizer_act = Quantizer(**quant_params_act)

    @override
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        q_a = self.quantizer_input_a(a)
        q_b = self.quantizer_input_b(b)

        y = super().forward(q_a, q_b)

        if self.activation:
            y = self.activation(y)

        return self.quantizer_act(y)
