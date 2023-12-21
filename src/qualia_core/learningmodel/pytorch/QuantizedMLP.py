from __future__ import annotations

import math
import sys
from collections import OrderedDict

from torch import nn

from qualia_core.learningmodel.pytorch.layers.quantized_layers import QuantizedIdentity, QuantizedLinear
from qualia_core.learningmodel.pytorch.LearningModelPyTorch import LearningModelPyTorch
from qualia_core.typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from qualia_core.typing import QuantizationConfigDict

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class QuantizedMLP(LearningModelPyTorch):
    def __init__(self,
                 input_shape: tuple[int, ...],
                 output_shape: tuple[int, ...],
                 units: list[int],
                 quant_params: QuantizationConfigDict) -> None:
        super().__init__(input_shape=input_shape, output_shape=output_shape)

        self.input_shape = input_shape
        self.output_shape = output_shape

        layers: OrderedDict[str, nn.Module] = OrderedDict()

        layers['identity1'] = QuantizedIdentity(quant_params=quant_params)
        layers['flatten1'] = nn.Flatten()

        i = 1
        for in_units, out_units in zip([math.prod(input_shape), *units], units[:-1]):
            layers[f'fc{i}'] = QuantizedLinear(in_units,
                                          out_units,
                                          quant_params=quant_params,
                                          activation=nn.ReLU())
            i += 1

        layers[f'fc{i}'] = QuantizedLinear(units[-1] if len(units) > 1 else math.prod(input_shape),
                                      output_shape[0],
                                      quant_params=quant_params)

        self.layers = nn.ModuleDict(layers)

    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        """Forward calls each of the SCNN :attr:`layers` sequentially.

        :param input: Input tensor
        :return: Output tensor
        """
        x = input
        for layer in self.layers:
            x = self.layers[layer](x)

        return x
