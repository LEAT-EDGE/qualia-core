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

class TorchVisionModelTransforms(DataAugmentationPyTorch):
    def __init__(self,  # noqa: PLR0913
                 weights_category: str,
                 weights: str = 'DEFAULT',
                 evaluate: bool = False,  # noqa: FBT001, FBT002
                 before: bool = False,  # noqa: FBT001, FBT002
                 after: bool = True) -> None:  # noqa: FBT001, FBT002
        super().__init__(evaluate=evaluate, before=before, after=after)

        from torchvision import models  # type: ignore[import-untyped]

        weights_enum = getattr(models, weights_category)
        weights_module = getattr(weights_enum, weights)

        self.transforms = weights_module.transforms()

    def apply(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trans_x = self.transforms(x)

        return trans_x, y

    @override
    def __call__(self, data: tuple[torch.Tensor, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        return self.apply(*data)
