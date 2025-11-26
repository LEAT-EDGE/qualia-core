from __future__ import annotations

import sys
from typing import Literal

import torch

from .DataAugmentationPyTorch import DataAugmentationPyTorch

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override


class ExponentialNoise(DataAugmentationPyTorch):

    def __init__(self,  # noqa: PLR0913
                 rate: float = 0.375,
                 round_mode: Literal['', 'floor', 'ceil', 'nearest', 'trunc'] = '',
                 dims: int = 1,
                 evaluate: bool = False,  # noqa: FBT001, FBT002
                 before: bool = False,  # noqa: FBT001, FBT002
                 after: bool = True) -> None:  # noqa: FBT001, FBT002
        super().__init__(evaluate=evaluate, before=before, after=after)
        self.rate = float(rate)
        self.round_mode = round_mode
        self.dims = dims

        self.noisedist = None

    def apply(self, x: torch.Tensor, y: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if self.noisedist is None:  # Lazy init so that CUDA is only used when module is actually called
            self.noisedist = torch.distributions.exponential.Exponential(torch.tensor(self.rate, device=device))

        noise = self.noisedist.sample(x.shape)

        if self.round_mode == 'floor':
            noise = noise.floor()
        elif self.round_mode == 'ceil':
            noise = noise.ceil()
        elif self.round_mode == 'nearest':
            noise = noise.round()
        elif self.round_mode == 'trunc':
            noise = noise.trunc()

        noisy_x = x + noise

        return noisy_x, y

    @override
    def __call__(self, data: tuple[torch.Tensor, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        return self.apply(*data, device=device)
