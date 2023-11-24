
from __future__ import annotations

import sys

from qualia_core.typing import TYPE_CHECKING

from .DataAugmentationPyTorch import DataAugmentationPyTorch

if TYPE_CHECKING:
    import torch  # noqa: TCH002

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class HorizontalFlip(DataAugmentationPyTorch):
    def __init__(self,
                 p: float = 0.5,
                 evaluate: bool = False,  # noqa: FBT001, FBT002
                 before: bool = False,  # noqa: FBT001, FBT002
                 after: bool = True) -> None:  # noqa: FBT001, FBT002
        import torchvision.transforms  # type: ignore[import-untyped]

        super().__init__(evaluate=evaluate, before=before, after=after)
        self.randomhorizontalflip = torchvision.transforms.RandomHorizontalFlip(p=p)

    def apply(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flipped_x = self.randomhorizontalflip(x)

        return flipped_x, y

    @override
    def __call__(self, data: tuple[torch.Tensor, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        return self.apply(*data)
