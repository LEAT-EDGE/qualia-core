from __future__ import annotations

import sys

from qualia_core.typing import TYPE_CHECKING

from .layers.SampleNorm import SampleNorm
from .LearningModelPyTorch import LearningModelPyTorch
from .ResNetStride import BasicBlockBuilder, ResNetStride

if TYPE_CHECKING:
    import torch  # noqa: TCH002

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class ResNetSampleNorm(LearningModelPyTorch):
    def __init__(self,  # noqa: PLR0913
                 input_shape: tuple[int, ...],
                 output_shape: tuple[int, ...],
                 filters: list[int],
                 kernel_sizes: list[int],
                 pool_sizes: list[int],

                 num_blocks: list[int],
                 strides: list[int],
                 paddings: list[int],
                 prepool: int = 1,
                 postpool: str = 'max',
                 batch_norm: bool = False,  # noqa: FBT001, FBT002
                 bn_momentum: float = 0.1,
                 force_projection_with_stride: bool = True,  # noqa: FBT001, FBT002

                 dims: int = 1,
                 basicblockbuilder: BasicBlockBuilder | None = None,
                 samplenorm: str = 'minmax') -> None:
        super().__init__(input_shape=input_shape, output_shape=output_shape)

        self.samplenorm = SampleNorm(norm=samplenorm)
        self.resnet = ResNetStride(input_shape=input_shape,
                         output_shape=output_shape,
                         filters=filters,
                         kernel_sizes=kernel_sizes,
                         pool_sizes=pool_sizes,
                         num_blocks=num_blocks,
                         strides=strides,
                         paddings=paddings,
                         prepool=prepool,
                         postpool=postpool,
                         batch_norm=batch_norm,
                         bn_momentum=bn_momentum,
                         force_projection_with_stride=force_projection_with_stride,
                         dims=dims,
                         basicblockbuilder=basicblockbuilder)

    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        x = self.samplenorm(x)
        return self.resnet(x)
