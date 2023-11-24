
from __future__ import annotations

import sys

import torch

from .DataAugmentationPyTorch import DataAugmentationPyTorch

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class GaussianNoise(DataAugmentationPyTorch):

    def __init__(self,
                 sigma: float = 0.375,
                 evaluate: bool = False,  # noqa: FBT001, FBT002
                 before: bool = False,  # noqa: FBT001, FBT002
                 after: bool = True) -> None:  # noqa: FBT001, FBT002
        super().__init__(evaluate=evaluate, before=before, after=after)
        self.sigma = sigma

        self.noisedist = None

    def apply(self, x: torch.Tensor, y: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if self.noisedist is None: # Lazy init so that CUDA is only used when module is actually called
            self.noisedist = torch.distributions.normal.Normal(torch.tensor(0.0, device=device),
                                                               torch.tensor(self.sigma, device=device))

        noise = self.noisedist.sample(x.shape[-1:]) # Generate noise for each time sample

        noisy_x = x + noise

        return noisy_x, y

    @override
    def __call__(self, data: tuple[torch.Tensor, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        return self.apply(*data, device=device)
