import sys

import torch
from torch import nn

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class GlobalSumPool2d(nn.Module):
    def __init__(self) -> None:
        self.call_super_init = True # Support multiple inheritance from nn.Module
        super().__init__()

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=(-2, -1))
