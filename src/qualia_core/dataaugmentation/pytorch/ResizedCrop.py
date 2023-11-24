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


class ResizedCrop(DataAugmentationPyTorch):
    def __init__(self,  # noqa: PLR0913
                 size: list[int],
                 scale: list[float] | None = None,
                 ratio: list[float] | None = None,
                 antialias: bool = True,  # noqa: FBT001, FBT002
                 evaluate: bool = False,  # noqa: FBT001, FBT002
                 before: bool = False,  # noqa: FBT001, FBT002
                 after: bool = True) -> None:  # noqa: FBT001, FBT002
        import torchvision.transforms  # type: ignore[import-untyped]

        super().__init__(evaluate=evaluate, before=before, after=after)

        scale = scale if scale is not None else [0.08, 1.0]
        ratio = ratio if ratio is not None else [0.75, 4/3]
        self.randomresizedcrop = torchvision.transforms.RandomResizedCrop(size=size,
                                                                          scale=tuple(scale),
                                                                          ratio=tuple(ratio),
                                                                          antialias=antialias)

    def apply(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        resizedcropped_x = self.randomresizedcrop(x)

        return resizedcropped_x, y

    @override
    def __call__(self, data: tuple[torch.Tensor, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        return self.apply(*data)
