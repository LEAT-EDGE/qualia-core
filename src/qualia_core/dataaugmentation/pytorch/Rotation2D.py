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

class Rotation2D(DataAugmentationPyTorch):

    def __init__(self,
                 angle: list[int] | None = None,
                 evaluate: bool = False,  # noqa: FBT001, FBT002
                 before: bool = False,  # noqa: FBT001, FBT002
                 after: bool = True) -> None:  # noqa: FBT001, FBT002
        import torchvision.transforms  # type: ignore[import-untyped]

        super().__init__(evaluate=evaluate, before=before, after=after)

        angle = angle if angle is not None else [-45, 45]
        self.randomrotation = torchvision.transforms.RandomRotation(angle)

    def apply(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rotated_x = self.randomrotation(x)

        return rotated_x, y

    @override
    def __call__(self, data: tuple[torch.Tensor, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        return self.apply(*data)
