from __future__ import annotations

import sys

from qualia_core.typing import TYPE_CHECKING, RecursiveConfigDict

from .DataAugmentationPyTorch import DataAugmentationPyTorch

if TYPE_CHECKING:
    import torch  # noqa: TCH002

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override


class MFCC(DataAugmentationPyTorch):
    def __init__(self,  # noqa: PLR0913, PLR0917
                 sample_rate: int,
                 n_mfcc: int,
                 dct_type: int = 2,
                 norm: str = 'ortho',
                 log_mels: bool = False,  # noqa: FBT002, FBT001
                 melkwargs: RecursiveConfigDict | None = None,
                 dims: int = 1,
                 evaluate: bool = False,  # noqa: FBT001, FBT002
                 before: bool = False,  # noqa: FBT001, FBT002
                 after: bool = True) -> None:  # noqa: FBT001, FBT002
        import torchaudio  # type: ignore[import-untyped]

        super().__init__(evaluate=evaluate, before=before, after=after)

        self.mfcc_transform = torchaudio.transforms.MFCC(sample_rate=sample_rate,
                                                        n_mfcc=n_mfcc,
                                                        dct_type=dct_type,
                                                        norm=norm,
                                                        log_mels=log_mels,
                                                        melkwargs=melkwargs)
        self.__dims = dims

    def apply(self, x: torch.Tensor, y: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mfcc_transform.dct_mat.device != device:
            self.mfcc_transform = self.mfcc_transform.to(device)
        x = self.mfcc_transform(x.squeeze(1))
        if self.__dims == 2:  # noqa: PLR2004
            x = x.swapaxes(1, 2)
            x = x.unsqueeze(1)
        return x, y  # Only handle single channel

    @override
    def __call__(self, data: tuple[torch.Tensor, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        return self.apply(*data, device=device)
