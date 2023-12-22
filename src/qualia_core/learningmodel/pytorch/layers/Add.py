import sys

import torch
import torch.nn.quantized
from torch import nn

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class Add(nn.Module):
    def __init__(self) -> None:
        self.call_super_init = True # Support multiple inheritance from nn.Module
        super().__init__()
        self.functional = torch.nn.quantized.FloatFunctional()

    @override
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.functional.add(a, b)

    @override
    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return super().__call__(a, b)
