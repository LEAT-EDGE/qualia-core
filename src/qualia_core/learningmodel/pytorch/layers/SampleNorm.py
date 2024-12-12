from __future__ import annotations

import logging
import sys

import torch
from torch import nn

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)

class SampleNorm(nn.Module):
    def __init__(self, norm: str) -> None:
        self.call_super_init = True # Support multiple inheritance from nn.Module
        super().__init__()

        if norm not in ['z', 'minmax']:
            logger.error('Unsupported mode %s, supported modes: z, minmax', norm)
            raise ValueError
        self.norm = norm

    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input

        if len(x.shape) != 3:  # noqa: PLR2004
            logger.error('Only 1D data with shape [N, C, S] is supported')
            raise ValueError

        if self.norm == 'z':
            x -= x.mean(dim=-1, keepdim=True)
            x /= x.std(dim=-1, keepdim=True)
        elif self.norm == 'minmax':
            x -= x.min(dim=-1, keepdim=True).values
            x /= x.max(dim=-1, keepdim=True).values

        return x

