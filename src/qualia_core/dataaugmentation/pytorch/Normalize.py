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

class Normalize(DataAugmentationPyTorch):
    def __init__(self,  # noqa: PLR0913
                 mean: list[int],
                 std: list[int],
                 evaluate: bool = False,  # noqa: FBT001, FBT002
                 before: bool = False,  # noqa: FBT001, FBT002
                 after: bool = True) -> None:  # noqa: FBT001, FBT002
        import torchvision.transforms  # type: ignore[import-untyped]

        super().__init__(evaluate=evaluate, before=before, after=after)

        self.normalize = torchvision.transforms.Normalize(mean=mean, std=std)

    def apply(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_x = self.normalize(x)

        return normalized_x, y

    @override
    def __call__(self, data: tuple[torch.Tensor, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        return self.apply(*data)
