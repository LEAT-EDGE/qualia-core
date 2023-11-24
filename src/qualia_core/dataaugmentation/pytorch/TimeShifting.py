from __future__ import annotations

import sys
from typing import cast

import torch

from .DataAugmentationPyTorch import DataAugmentationPyTorch

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override


class TimeShifting(DataAugmentationPyTorch):
    def __init__(self,
                 alpha: float = 1.0,
                 evaluate: bool = False,  # noqa: FBT001, FBT002
                 before: bool = False,  # noqa: FBT001, FBT002
                 after: bool = True) -> None:  # noqa: FBT001, FBT002
        super().__init__(evaluate=evaluate, before=before, after=after)

        self.alpha = alpha

        self.dist = None

    def apply(self, x: torch.Tensor, y: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if self.dist is None:
            self.dist = torch.distributions.beta.Beta(torch.tensor(self.alpha, device=device),
                                                      torch.tensor(self.alpha, device=device))

        ratio = self.dist.sample()

        shift = (x.shape[-1] * ratio).int()

        shifted_x = torch.roll(x, cast(int, shift.item()), dims=-1)

        return shifted_x, y

    @override
    def __call__(self, data: tuple[torch.Tensor, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        return self.apply(*data, device=device)
