from __future__ import annotations

import sys

import torch

from .DataAugmentationPyTorch import DataAugmentationPyTorch

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class IntToFloat32(DataAugmentationPyTorch):
    def __init__(self,
                 scale: bool,  # noqa: FBT001
                 evaluate: bool = False,  # noqa: FBT001, FBT002
                 before: bool = False,  # noqa: FBT001, FBT002
                 after: bool = True) -> None:  # noqa: FBT001, FBT002
        super().__init__(evaluate=evaluate, before=before, after=after)
        self.__scale = scale

    def apply(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        float_x = x.to(torch.float32)
        if self.__scale:
            iinfo = torch.iinfo(x.dtype)
            return x / iinfo.max, y
        return float_x, y

    @override
    def __call__(self, data: tuple[torch.Tensor, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        return self.apply(*data)
