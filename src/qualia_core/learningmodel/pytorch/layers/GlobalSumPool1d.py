import sys

import torch.nn as nn
import torch

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class GlobalSumPool1d(nn.Module):
    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=(-1))
