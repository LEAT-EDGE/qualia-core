from __future__ import annotations

import sys

import torch

from qualia_core.typing import TYPE_CHECKING

from .DataAugmentationPyTorch import DataAugmentationPyTorch

if TYPE_CHECKING:
    from torchvision.transforms import AutoAugmentPolicy  # type: ignore[import-untyped] # noqa: TCH002

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class AutoAugment(DataAugmentationPyTorch):
    def __init__(self,
                 policy: str,
                 evaluate: bool = False,  # noqa: FBT001, FBT002
                 before: bool = False,  # noqa: FBT001, FBT002
                 after: bool = True) -> None:  # noqa: FBT001, FBT002
        import torchvision.transforms  # type: ignore[import-untyped]

        super().__init__(evaluate=evaluate, before=before, after=after)
        autoaugmentpolicy: AutoAugmentPolicy = getattr(torchvision.transforms.AutoAugmentPolicy, policy)
        self.autoaugment = torchvision.transforms.AutoAugment(autoaugmentpolicy)

    def apply(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        torch.use_deterministic_algorithms(mode=False)
        augmented_x = self.autoaugment(x)
        torch.use_deterministic_algorithms(mode=True)

        return augmented_x, y

    @override
    def __call__(self, data: tuple[torch.Tensor, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        return self.apply(*data)
