from __future__ import annotations

import sys

import torch

from .DataAugmentationPyTorch import DataAugmentationPyTorch

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override


class Cutout1D(DataAugmentationPyTorch):
    def __init__(self,
                 length_sigma: float = 1.0,
                 evaluate: bool = False,  # noqa: FBT001, FBT002
                 before: bool = False,  # noqa: FBT001, FBT002
                 after: bool = True) -> None:  # noqa: FBT001, FBT002
        super().__init__(evaluate=evaluate, before=before, after=after)

        self.length_sigma = length_sigma

        self.offset_dist = None
        self.length_dist = None

    def apply(self, x: torch.Tensor, y: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if self.offset_dist is None:
            self.offset_dist = torch.distributions.uniform.Uniform(torch.tensor(0.0, device=device),
                                                       torch.tensor(1.0, device=device))
        if self.length_dist is None:
            self.length_dist = torch.distributions.normal.Normal(torch.tensor(0.0, device=device),
                                                                 torch.tensor(self.length_sigma, device=device))

        offset = (self.offset_dist.sample() * x.size(-1)).long()
        length = (self.length_dist.sample().abs() * (x.size(-1) - offset)).long()

        # Prevent setting the entire tensor to 0 which is useless
        if offset < 1 and length > x.size(-1):
            return x, y

        x[..., offset:offset + length] = 0.0

        return x, y

    @override
    def __call__(self, data: tuple[torch.Tensor, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        return self.apply(*data, device=device)
