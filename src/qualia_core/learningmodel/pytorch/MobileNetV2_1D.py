from __future__ import annotations

import logging
import sys

import torch
from torch import nn

from qualia_core.learningmodel.pytorch.LearningModelPyTorch import LearningModelPyTorch
from qualia_core.learningmodel.pytorch.layers import Add
from qualia_core.typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)


class ConvNormActivation(nn.Sequential):
    def __init__(
        self,
        layers_t: ModuleType,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        batch_norm: bool = True,  # noqa: FBT001, FBT002
        bn_momentum: float = 0.1,
        activation: nn.Module | None = None,
    ) -> None:
        layers: list[nn.Module] = [
            layers_t.Conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=not batch_norm,
            )
        ]
        if batch_norm:
            layers.append(layers_t.BatchNorm(out_channels, momentum=bn_momentum))
        if activation is not None:
            layers.append(activation)
        super().__init__(*layers)


class InvertedResidual1D(nn.Module):
    def __init__(
        self,
        layers_t: ModuleType,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        expansion_factor: int,
        batch_norm: bool,  # noqa: FBT001
        bn_momentum: float,
    ) -> None:
        super().__init__()
        hidden_channels = in_channels * expansion_factor
        self.use_residual = stride == 1 and in_channels == out_channels

        layers: list[nn.Module] = []
        if expansion_factor != 1:
            layers.append(
                ConvNormActivation(
                    layers_t,
                    in_channels,
                    hidden_channels,
                    kernel_size=1,
                    batch_norm=batch_norm,
                    bn_momentum=bn_momentum,
                    activation=nn.ReLU6(),
                )
            )

        layers.extend(
            [
                ConvNormActivation(
                    layers_t,
                    hidden_channels,
                    hidden_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=hidden_channels,
                    batch_norm=batch_norm,
                    bn_momentum=bn_momentum,
                    activation=nn.ReLU6(),
                ),
                ConvNormActivation(
                    layers_t,
                    hidden_channels,
                    out_channels,
                    kernel_size=1,
                    batch_norm=batch_norm,
                    bn_momentum=bn_momentum,
                    activation=None,
                ),
            ]
        )
        self.layers = nn.Sequential(*layers)
        if self.use_residual:
            self.add = Add()

    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        out = self.layers(input)
        if self.use_residual:
            out = self.add(out, input)
        return out


class MobileNetV2_1D(LearningModelPyTorch):
    def __init__(
        self,
        input_shape: tuple[int, ...],
        output_shape: tuple[int, ...],
        stem_channels: int,
        stem_kernel_size: int,
        stem_stride: int,
        stem_padding: int,
        block_out_channels: list[int],
        block_repeats: list[int],
        block_strides: list[int],
        block_expansion_factors: list[int],
        block_kernel_sizes: list[int] | None = None,
        block_paddings: list[int] | None = None,
        head_channels: int = 128,
        dropout: float = 0.0,
        batch_norm: bool = True,  # noqa: FBT001, FBT002
        bn_momentum: float = 0.1,
        dims: int = 1,
    ) -> None:
        super().__init__(input_shape=input_shape, output_shape=output_shape)

        if dims != 1:
            logger.error("MobileNetV2_1D only supports dims=1, got: %s", dims)
            raise ValueError

        import qualia_core.learningmodel.pytorch.layers.layers1d as layers_t

        n_blocks = len(block_out_channels)
        if not (len(block_repeats) == len(block_strides) == len(block_expansion_factors) == n_blocks):
            raise ValueError("block_out_channels, block_repeats, block_strides, and block_expansion_factors must have the same length")
        if block_kernel_sizes is None:
            block_kernel_sizes = [3] * n_blocks
        if block_paddings is None:
            block_paddings = [1] * n_blocks
        if not (len(block_kernel_sizes) == len(block_paddings) == n_blocks):
            raise ValueError("block_kernel_sizes and block_paddings must match block_out_channels length")

        in_channels = input_shape[-1]
        self.stem = ConvNormActivation(
            layers_t,
            in_channels,
            stem_channels,
            kernel_size=stem_kernel_size,
            stride=stem_stride,
            padding=stem_padding,
            batch_norm=batch_norm,
            bn_momentum=bn_momentum,
            activation=nn.ReLU6(),
        )

        blocks: list[nn.Module] = []
        current_channels = stem_channels
        for out_channels, repeats, stride, expansion, kernel_size, padding in zip(
            block_out_channels,
            block_repeats,
            block_strides,
            block_expansion_factors,
            block_kernel_sizes,
            block_paddings,
        ):
            for repeat_index in range(repeats):
                blocks.append(
                    InvertedResidual1D(
                        layers_t=layers_t,
                        in_channels=current_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride if repeat_index == 0 else 1,
                        padding=padding,
                        expansion_factor=expansion,
                        batch_norm=batch_norm,
                        bn_momentum=bn_momentum,
                    )
                )
                current_channels = out_channels
        self.blocks = nn.Sequential(*blocks)

        self.head = ConvNormActivation(
            layers_t,
            current_channels,
            head_channels,
            kernel_size=1,
            batch_norm=batch_norm,
            bn_momentum=bn_momentum,
            activation=nn.ReLU6(),
        )
        self.postpool = layers_t.AdaptiveAvgPool(1)
        self.flatten = nn.Flatten()
        if dropout:
            self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(head_channels, output_shape[0])

    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        out = self.stem(input)
        out = self.blocks(out)
        out = self.head(out)
        out = self.postpool(out)
        out = self.flatten(out)
        if hasattr(self, "dropout"):
            out = self.dropout(out)
        return self.linear(out)
