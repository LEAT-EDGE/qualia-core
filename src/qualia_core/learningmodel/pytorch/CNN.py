from __future__ import annotations

import logging
import math
import sys
from collections import OrderedDict

import numpy as np
from torch import nn

from qualia_core.typing import TYPE_CHECKING

from .layers import layers1d, layers2d
from .LearningModelPyTorch import LearningModelPyTorch

if TYPE_CHECKING:
    from types import ModuleType

    import torch

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)


class CNN(LearningModelPyTorch):
    def __init__(self, # noqa: PLR0913, PLR0915, PLR0912, C901
                 input_shape: tuple[int, ...],
                 output_shape: tuple[int, ...],

                 filters: list[int],
                 kernel_sizes: list[int],
                 paddings: list[int],
                 strides: list[int],
                 dropouts: float | list[float],
                 pool_sizes: list[int],
                 fc_units: list[int],
                 separables: list[bool] | None = None,
                 batch_norm: bool = False,  # noqa: FBT001, FBT002
                 prepool: int | list[int] = 1,
                 postpool: int | list[int] = 1,

                 gsp: bool = False,  # noqa: FBT001, FBT002
                 dims: int = 1) -> None:
        super().__init__(input_shape=input_shape, output_shape=output_shape)

        layers_t: ModuleType

        if dims == 1:
            layers_t = layers1d
        elif dims == 2:  # noqa: PLR2004
            layers_t = layers2d
        else:
            logger.error('Only dims=1 or dims=2 supported, got: %s', dims)
            raise ValueError

        # Backward compatibility for config not defining dropout as a list
        dropout_list = [dropouts] * (len(filters) + len(fc_units)) if not isinstance(dropouts, list) else dropouts

        if separables is None:
            separables = [False] * len(filters)

        layers: OrderedDict[str, nn.Module] = OrderedDict()

        if (math.prod(prepool) if isinstance(prepool, list) else prepool) > 1:
            layers['prepool'] = layers_t.AvgPool(tuple(prepool) if isinstance(prepool, list) else prepool)

        i = 1
        for (in_filters,
             out_filters,
             kernel,
             pool_size,
             padding,
             stride,
             dropout,
             separable) in zip([input_shape[-1], *filters],
                               filters,
                               kernel_sizes,
                               pool_sizes,
                               paddings,
                               strides,
                               dropout_list,
                               separables):
            if separable:
                layers[f'conv{i}_dw'] = layers_t.Conv(in_channels=in_filters,
                                                      out_channels=in_filters,
                                                      kernel_size=kernel,
                                                      padding=padding,
                                                      stride=stride,
                                                      groups=in_filters,
                                                      bias=not batch_norm)

                if batch_norm:
                    layers[f'bn{i}_dw'] = layers_t.BatchNorm(out_filters)

                layers[f'relu{i}_dw'] = nn.ReLU()

                layers[f'conv{i}_pw'] = layers_t.Conv(in_channels=in_filters,
                                                      out_channels=out_filters,
                                                      kernel_size=1,
                                                      padding=0,
                                                      stride=1,
                                                      bias=not batch_norm)

                if batch_norm:
                    layers[f'bn{i}_pw'] = layers_t.BatchNorm(out_filters)

                layers[f'relu{i}_pw'] = nn.ReLU()
            else:

                layers[f'conv{i}'] = layers_t.Conv(in_channels=in_filters,
                                                   out_channels=out_filters,
                                                   kernel_size=kernel,
                                                   padding=padding,
                                                   stride=stride,
                                                   bias=not batch_norm)

                if batch_norm:
                    layers[f'bn{i}'] = layers_t.BatchNorm(out_filters)

                layers[f'relu{i}'] = nn.ReLU()

            if dropout:
                layers[f'dropout{i}'] = nn.Dropout(dropout)
            if pool_size:
                layers[f'maxpool{i}'] = layers_t.MaxPool(pool_size)

            i += 1

        if (math.prod(postpool) if isinstance(postpool, list) else postpool) > 1:
            layers['postpool'] = layers_t.AvgPool(tuple(postpool) if isinstance(postpool, list) else postpool)

        if gsp:
            layers[f'conv{i}'] = layers_t.Conv(in_channels=filters[-1],
                                               out_channels=output_shape[0],
                                               kernel_size=1,
                                               padding=0,
                                               stride=1,
                                               bias=True)
            layers[f'relu{i}'] = nn.ReLU()
            layers['gsp'] = layers_t.GlobalSumPool()
        else:
            layers['flatten'] = nn.Flatten()

            in_features = np.array(input_shape[:-1]) // np.array(prepool)
            for _, kernel, pool, padding, stride in zip(filters, kernel_sizes, pool_sizes, paddings, strides):
                in_features += np.array(padding) * 2
                in_features -= (np.array(kernel) - 1)
                in_features = np.ceil(in_features / stride).astype(int)
                if pool:
                    in_features = in_features // pool
            in_features = in_features // np.array(postpool)
            in_features = in_features.prod()
            in_features *= filters[-1]

            j = 1
            for in_units, out_units, dropout in zip((in_features, *fc_units), fc_units, dropout_list[len(filters):]):
                layers[f'fc{j}'] = nn.Linear(in_units, out_units)
                layers[f'relu{i}'] = nn.ReLU()
                if dropout:
                    layers[f'dropout{i}'] = nn.Dropout(dropout)
                i += 1
                j += 1

            layers[f'fc{j}'] = nn.Linear(fc_units[-1] if len(fc_units) > 0 else in_features, output_shape[0])

        self.layers = nn.ModuleDict(layers)

    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward calls each of the SCNN :attr:`layers` sequentially.

        :param input: Input tensor
        :return: Output tensor
        """
        x = input
        for layer in self.layers:
            x = self.layers[layer](x)

        return x
