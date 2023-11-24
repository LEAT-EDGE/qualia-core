from __future__ import annotations

import math
import sys

import torch

from .DataAugmentationPyTorch import DataAugmentationPyTorch

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override


class TimeWarping(DataAugmentationPyTorch):
    def __init__(self,
                 sigma: float = 0.25,
                 evaluate: bool = False,  # noqa: FBT001, FBT002
                 before: bool = False,  # noqa: FBT001, FBT002
                 after: bool = True) -> None:  # noqa: FBT001, FBT002
        super().__init__(evaluate=evaluate, before=before, after=after)

        self.sigma = sigma

        self.dist = None

    def apply(self, x: torch.Tensor, y: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if self.dist is None:
            self.dist = torch.distributions.normal.Normal(torch.tensor(1.0, device=device),
                                                          torch.tensor(self.sigma, device=device))

        scale_factor = self.dist.sample()

        # Don't apply time warping if it would result in a null or negative length
        if scale_factor * x.shape[-1] < 1:
            return x, y

        warped_x = torch.nn.functional.interpolate(x,
                                                   scale_factor=(scale_factor, ),
                                                   mode='linear',
                                                   recompute_scale_factor=True)

        warped_x = torch.tile(warped_x, (1, 1, math.ceil(x.shape[-1] / warped_x.shape[-1])))
        warped_x = warped_x[:,:,:x.shape[-1]]

        return warped_x, y

    @override
    def __call__(self, data: tuple[torch.Tensor, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        return self.apply(*data, device=device)
