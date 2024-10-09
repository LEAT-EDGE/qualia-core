from __future__ import annotations

import logging
import sys
from typing import Protocol

import numpy as np
import torch
from torch import nn

from qualia_core.learningmodel.pytorch.layers import Add
from qualia_core.typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType  # noqa: TCH003

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)

class BasicBlockBuilder(Protocol):
    """Signature for basicblockbuilder.

    Used to bind hyperparameters constant across all the ResNet blocks.
    """

    def __call__(self,
                 in_planes: int,
                 planes: int,
                 kernel_size: int,
                 stride: int,
                 padding: int) -> BasicBlock:
        """Build a :class:`BasicBlock`.

        :param in_planes: Number of input channels
        :param planes: Number of filters (i.e., output channels) in the main branch Conv layers
        :param kernel_size: ``kernel_size`` for the main branch Conv layers
        :param stride: ``kernel_size`` for the MaxPool layers, no MaxPool layer added if `1`
        :param padding: Padding for the main branch Conv layers
        :return: A :class:`BasicBlock`
        """
        ...

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,  # noqa: PLR0913
                 layers_t: ModuleType,
                 in_planes: int,
                 planes: int,
                 kernel_size: int,
                 stride: int,
                 padding: int,
                 batch_norm: bool,  # noqa: FBT001
                 bn_momentum: float,
                 force_projection_with_stride: bool) -> None:  # noqa: FBT001
        super().__init__()

        self.batch_norm = batch_norm
        self.force_projection_with_stride = force_projection_with_stride
        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride

        self.conv1 = layers_t.Conv(in_planes,
                                   planes,
                                   kernel_size=kernel_size,
                                   stride=1,
                                   padding=padding,
                                   bias=not batch_norm)
        if batch_norm:
            self.bn1 = layers_t.BatchNorm(planes, momentum=bn_momentum)
        if self.stride != 1:
            self.pool1 = layers_t.MaxPool(stride)
        self.relu1 = nn.ReLU()

        self.conv2 = layers_t.Conv(planes, planes, kernel_size=kernel_size, stride=1, padding=padding, bias=not batch_norm)
        if batch_norm:
            self.bn2 = layers_t.BatchNorm(planes, momentum=bn_momentum)

        if self.stride != 1:
            self.spool = layers_t.MaxPool(stride)
        if self.in_planes != self.expansion * self.planes or self.force_projection_with_stride and self.stride != 1:
            self.sconv = layers_t.Conv(in_planes, self.expansion * planes, kernel_size=1, stride=1, bias=not batch_norm)
            if batch_norm:
                self.sbn = layers_t.BatchNorm(
                    self.expansion*planes, momentum=bn_momentum)

        self.add = Add()
        self.relu = nn.ReLU()

    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input

        out = self.conv1(x)

        if self.batch_norm:
            out = self.bn1(out)
        if self.stride != 1:
            out = self.pool1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        if self.batch_norm:
            out = self.bn2(out)


        # shortcut
        tmp = x

        if self.in_planes != self.expansion * self.planes or self.force_projection_with_stride and self.stride != 1:
            tmp = self.sconv(tmp)

            if self.batch_norm:
                tmp = self.sbn(tmp)

        if self.stride != 1:
            tmp = self.spool(tmp)

        out = self.add(out, tmp)
        return self.relu(out)

class ResNet(nn.Module):
    def __init__(self,  # noqa: PLR0913, C901
                 input_shape: tuple[int, ...],
                 output_shape: tuple[int, ...],
                 filters: list[int],
                 kernel_sizes: list[int],

                 num_blocks: list[int],
                 strides: list[int],
                 paddings: list[int],
                 prepool: int = 1,
                 postpool: str = 'max',
                 batch_norm: bool = False,  # noqa: FBT001, FBT002
                 bn_momentum: float = 0.1,
                 force_projection_with_stride: bool = True,  # noqa: FBT001, FBT002

                 dims: int = 1,
                 basicblockbuilder: BasicBlockBuilder | None = None) -> None:
        super().__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape

        if dims == 1:
            import qualia_core.learningmodel.pytorch.layers1d as layers_t
        elif dims == 2:  # noqa: PLR2004
            import qualia_core.learningmodel.pytorch.layers2d as layers_t
        else:

            logger.error('Only dims=1 or dims=2 supported, got: %s', dims)
            raise ValueError

        if basicblockbuilder is None:

            def builder(in_planes: int,
                        planes: int,
                        kernel_size: int,
                        stride: int,
                        padding: int) -> BasicBlock:
                return BasicBlock(
                        layers_t=layers_t,
                        in_planes=in_planes,
                        planes=planes,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        batch_norm=batch_norm,
                        bn_momentum=bn_momentum,
                        force_projection_with_stride=force_projection_with_stride)
            basicblockbuilder = builder

        self.in_planes = filters[0]
        self.batch_norm = batch_norm
        self.num_blocks = num_blocks

        if prepool > 1:
            self.prepool = layers_t.AvgPool(prepool)

        self.conv1 = layers_t.Conv(input_shape[-1],
                                   filters[0],
                                   kernel_size=kernel_sizes[0],
                                   stride=strides[0],
                                   padding=paddings[0],
                                   bias=not batch_norm)
        if self.batch_norm:
            self.bn1 = layers_t.BatchNorm(self.in_planes, momentum=bn_momentum)
        self.relu1 = nn.ReLU()

        blocks: list[nn.Sequential] = []
        for planes, kernel_size, stride, padding, num_block in zip(filters[1:],
                                                                   kernel_sizes[1:],
                                                                   strides[1:],
                                                                   paddings[1:],
                                                                   num_blocks):
            blocks += [self._make_layer(basicblockbuilder, num_block, planes, kernel_size, stride, padding)]
        self.layers = nn.ModuleList(blocks)

        # GlobalMaxPool kernel_size computation
        self._fm_dims = np.array(input_shape[:-1]) // np.array(prepool)
        for _, kernel, stride, padding in zip(filters, kernel_sizes, strides, paddings):
            self._fm_dims += np.array(padding) * 2
            self._fm_dims -= (kernel - 1)
            self._fm_dims = np.floor(self._fm_dims / stride).astype(int)

        if postpool == 'avg':
            self.postpool = layers_t.AdaptiveAvgPool(1)
        elif postpool == 'max':
            self.postpool = layers_t.MaxPool(tuple(self._fm_dims))

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(self.in_planes*BasicBlock.expansion, output_shape[0])


    def _make_layer(self,  # noqa: PLR0913
                    basicblockbuilder: BasicBlockBuilder,
                    num_blocks: int,
                    planes: int,
                    kernel_size: int,
                    stride: int,
                    padding: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks-1)
        layers: list[BasicBlock] = []
        for s in strides:
            block = basicblockbuilder(in_planes=self.in_planes,
                                      planes=planes,
                                      kernel_size=kernel_size,
                                      stride=s,
                                      padding=padding)
            layers.append(block)
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        if hasattr(self, 'prepool'):
            x = self.prepool(x)

        out = self.conv1(x)
        if self.batch_norm:
            out = self.bn1(out)
        out = self.relu1(out)

        for i in range(len(self.layers)):
            out = self.layers[i](out)

        if hasattr(self, 'postpool'):
            out = self.postpool(out)

        out = self.flatten(out)
        return self.linear(out)
