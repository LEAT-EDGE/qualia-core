import math
from collections import OrderedDict
from typing import Union

import numpy as np
from torch import nn


class CNN(nn.Sequential):
    def __init__(self,
        input_shape,
        output_shape,

        filters: list=[6, 16, 120],
        kernel_sizes: list=[3, 3, 5],
        paddings: list=[0, 0, 0],
        strides: list=[1, 1, 1],
        batch_norm: bool=False,
        dropouts: Union[float, list[float]] = 0.0,
        pool_sizes: list=[2, 2, 0],
        fc_units: list=[84],
        prepool: int=1,
        postpool: int=1,

        gsp: bool=False,

        dims: int=1):

        self.input_shape = input_shape
        self.output_shape = output_shape

        if dims == 1:
            import qualia_core.learningmodel.pytorch.layers1d as layers_t
        elif dims == 2:
            import qualia_core.learningmodel.pytorch.layers2d as layers_t
        else:
            raise ValueError('Only dims=1 or dims=2 supported')

        # Backward compatibility for config not defining dropout as a list
        if not isinstance(dropouts, list):
            dropouts = [dropouts] * (len(filters) + len(fc_units))

        layers = OrderedDict()

        if not isinstance(prepool, int) and math.prod(prepool) > 1 or prepool > 1:
            layers['prepool'] = layers_t.AvgPool(prepool)

        layers['conv1'] = layers_t.Conv(in_channels=input_shape[-1], out_channels=filters[0], kernel_size=kernel_sizes[0], padding=paddings[0], stride=strides[0])

        if batch_norm:
            layers['bn1'] = layers_t.BatchNorm(filters[0])

        layers['relu1'] = nn.ReLU()

        if dropouts[0]:
            layers['dropout1'] = nn.Dropout(dropouts[0])
        if pool_sizes[0]:
            layers['maxpool1'] = layers_t.MaxPool(pool_sizes[0])

        i = 2
        for in_filters, out_filters, kernel, pool_size, padding, stride, dropout in zip(filters, filters[1:], kernel_sizes[1:],
                                                                                        pool_sizes[1:], paddings[1:], strides[1:],
                                                                                        dropouts[1:]):
            layers[f'conv{i}'] = layers_t.Conv(in_channels=in_filters, out_channels=out_filters, kernel_size=kernel, padding=padding, stride=stride)

            if batch_norm:
                layers[f'bn{i}'] = layers_t.BatchNorm(out_filters)

            layers[f'relu{i}'] = nn.ReLU()

            if dropout:
                layers[f'dropout{i}'] = nn.Dropout(dropout)
            if pool_size:
                layers[f'maxpool{i}'] = layers_t.MaxPool(pool_size)

            i += 1

        if not isinstance(postpool, int) and math.prod(postpool) > 1 or postpool > 1:
            layers['postpool'] = layers_t.AvgPool(postpool)

        if gsp:
            layers['gsp'] = layers_t.GlobalSumPool()
        else:
            layers['flatten'] = nn.Flatten()

            in_features = np.array(input_shape[:-1]) // np.array(prepool)
            for _, kernel, pool, padding, stride in zip(filters, kernel_sizes, pool_sizes, paddings, strides):
                in_features += np.array(padding) * 2
                in_features -= (kernel - 1)
                in_features = np.ceil(in_features / stride).astype(int)
                if pool:
                    in_features = in_features // pool
            in_features = in_features // np.array(postpool)
            in_features = in_features.prod()
            in_features *= filters[-1]

            j = 1
            for in_units, out_units, dropout in zip((in_features, *fc_units), fc_units, dropouts[len(filters):]):
                layers[f'fc{j}'] = nn.Linear(in_units, out_units)
                layers[f'relu{i}'] = nn.ReLU()
                if dropout:
                    layers[f'dropout{i}'] = nn.Dropout(dropout)
                i += 1
                j += 1

            layers[f'fc{j}'] = nn.Linear(fc_units[-1] if len(fc_units) > 0 else in_features, output_shape[0])

        super().__init__(layers)
