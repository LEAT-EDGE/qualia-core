from __future__ import annotations

import sys

import torch

from .DataAugmentationPyTorch import DataAugmentationPyTorch

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class Amplitude(DataAugmentationPyTorch):
    def __init__(self,
                 distribution: str = 'uniform',
                 low: float = 0.0,
                 high: float = 1.0,
                 sigma: float = 1.0,
                 evaluate: bool = False,  # noqa: FBT001, FBT002
                 before: bool = False,  # noqa: FBT001, FBT002
                 after: bool = True) -> None:  # noqa: FBT001, FBT002

        super().__init__(evaluate=evaluate, before=before, after=after)

        self._sigma = sigma
        self._low = low
        self._high = high

        if distribution == 'uniform':
            self._dist_f = self._uniform
        elif distribution  == 'normal':
            self._dist_f = self._normal
        else:
            raise ValueError

        self.dist = None

    def _normal(self, device: torch.device) -> torch.distributions.Distribution:
        return torch.distributions.normal.Normal(torch.tensor(1.0, device=device),
                                                 torch.tensor(self._sigma, device=device))

    def _uniform(self, device: torch.device) -> torch.distributions.Distribution:
        return torch.distributions.uniform.Uniform(torch.tensor(self._low, device=device),
                                                   torch.tensor(self._high, device=device))

    def apply(self, x: torch.Tensor, y: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if self.dist is None:
            self.dist = self._dist_f(device=device)

        scale_factor = self.dist.sample()

        amplified_x = x * scale_factor

        return amplified_x, y

    @override
    def __call__(self, data: tuple[torch.Tensor, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        return self.apply(*data, device=device)
