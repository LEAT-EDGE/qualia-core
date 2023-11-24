from __future__ import annotations

import sys

import torch
from torch import nn

from .DataAugmentationPyTorch import DataAugmentationPyTorch

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class Mixup(DataAugmentationPyTorch):
    def __init__(self,
                 alpha: float = 1.0,
                 evaluate: bool = False,  # noqa: FBT001, FBT002
                 before: bool = False,  # noqa: FBT001, FBT002
                 after: bool = True) -> None:  # noqa: FBT001, FBT002
        super().__init__(evaluate=evaluate, before=before, after=after)

        self.alpha = alpha
        self.crossentropyloss = nn.CrossEntropyLoss()

        self.lambdadist: torch.distributions.beta.Beta | None = None

    def __mix_criterion(self,  # noqa: PLR0913
                        criterion: nn.Module,
                        pred: torch.Tensor,
                        y_a: torch.Tensor,
                        y_b: torch.Tensor,
                        l: torch.Tensor) -> torch.Tensor:
        l = l.mean() # same for whole batch
        return l * criterion(pred, y_a) + (1 - l) * criterion(pred, y_b)

    def loss(self, pred: torch.Tensor, targets: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return self.__mix_criterion(self.crossentropyloss, pred, *targets)

    def apply(self,
              x: torch.Tensor,
              y: torch.Tensor,
              device: torch.device) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Return mixed inputs, pairs of targets, and lambda."""
        if self.lambdadist is None: # Lazy init so that CUDA is only used when module is actually called
            self.lambdadist = torch.distributions.beta.Beta(torch.tensor(self.alpha, device=device),
                                                            torch.tensor(self.alpha, device=device))

        l = self.lambdadist.sample()

        batch_size = x.shape[0]
        index = torch.randperm(batch_size)

        mixed_x = l * x + (1 - l) * x[index, :]
        y_a, y_b = y, y[index]

        return mixed_x, (y_a, y_b, l)

    @override
    def __call__(self,
                 data: tuple[torch.Tensor, torch.Tensor],
                 device: torch.device) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        return self.apply(*data, device=device)
