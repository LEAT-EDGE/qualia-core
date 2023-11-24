import math
from collections import OrderedDict
from typing import Optional, Union

import numpy as np
import torch
from qualia_core.learningmodel.pytorch.layers.quantized_layers import QuantizedIdentity, QuantizedLinear, QuantizedReLU
from qualia_core.typing import RecursiveConfigDict
from torch import nn


class QuantizedCNN(nn.Sequential):
    def __init__(self,
                 input_shape: tuple[int],
                 output_shape: tuple[int],

                 filters: list[int] = [6, 16, 120],
                 kernel_sizes: list[int] = [3, 3, 5],
                 paddings: list[int] = [0, 0, 0],
                 strides: list[int] = [1, 1, 1],
                 batch_norm: bool = False,
                 dropouts: Union[float, list[float]] = 0.0,
                 pool_sizes: list[int] = [2, 2, 0],
                 fc_units: list[int] = [84],
                 prepool: Union[int, list[int]] = 1,
                 postpool: Union[int, list[int]]=1,

                 dims: int=1,

                 quantize_linear: bool = True,
                 fused_relu: bool = True,
                 quant_params: Optional[RecursiveConfigDict] = None) -> None:

        self.input_shape = input_shape
        self.output_shape = output_shape

        if quant_params and 'bits' in quant_params :
            quant_params['bits'] = int(quant_params['bits']) # Force conversion from TOML int to plain Python int
            if quant_params['bits'] < 1:
                raise ValueError('bits must be set to a strictly positive integer')
        else :
            raise ValueError('bits must exist in quant_params conf')

        if dims == 1:
            import qualia_core.learningmodel.pytorch.layers1d as layers_t
        elif dims == 2:
            import qualia_core.learningmodel.pytorch.layers2d as layers_t
        else:
            raise ValueError('Only dims=1 or dims=2 supported')

        # Backward compatibility for config not defining dropout as a list
        if not isinstance(dropouts, list):
            dropouts = [dropouts] * (len(filters) + len(fc_units))

        layers: OrderedDict[str, nn.Module] = OrderedDict()

        layers['identity1'] = QuantizedIdentity(quant_params=quant_params)

        if isinstance(prepool, int) and prepool > 1:
            layers['prepool'] = layers_t.QuantizedAvgPool(prepool, quant_params=quant_params)
        elif not isinstance(prepool, int) and math.prod(prepool) > 1:
            layers['prepool'] = layers_t.QuantizedAvgPool(tuple(prepool), quant_params=quant_params)

        layers['conv1'] = layers_t.QuantizedConv(in_channels=input_shape[-1],
                                                 out_channels=filters[0],
                                                 kernel_size=kernel_sizes[0],
                                                 padding=paddings[0],
                                                 stride=strides[0],
                                                 quant_params=quant_params,
                                                 activation=nn.ReLU() if fused_relu and not batch_norm else None)

        if batch_norm:
            layers['bn1'] = layers_t.QuantizedBatchNorm(filters[0],
                                                        quant_params=quant_params,
                                                        activation=nn.ReLU() if fused_relu else None)

        if not fused_relu:
            layers['relu1'] = QuantizedReLU(quant_params=quant_params)

        if dropouts[0]:
            layers['dropout1'] = nn.Dropout(dropouts[0])
        if pool_sizes[0]:
            layers['maxpool1'] = layers_t.QuantizedMaxPool(pool_sizes[0], quant_params=quant_params)

        i = 2
        for in_filters, out_filters, kernel, pool_size, padding, stride, dropout in zip(filters, filters[1:], kernel_sizes[1:],
                                                                                        pool_sizes[1:], paddings[1:], strides[1:],
                                                                                        dropouts[1:]):
            layers[f'conv{i}'] = layers_t.QuantizedConv(in_channels=in_filters,
                                                        out_channels=out_filters,
                                                        kernel_size=kernel,
                                                        padding=padding,
                                                        stride=stride,
                                                        quant_params=quant_params,
                                                        activation=nn.ReLU() if fused_relu and not batch_norm else None)

            if batch_norm:
                layers[f'bn{i}'] = layers_t.QuantizedBatchNorm(out_filters,
                                                               quant_params=quant_params,
                                                               activation=nn.ReLU() if fused_relu else None)

            if not fused_relu:
                layers[f'relu{i}'] = QuantizedReLU(quant_params=quant_params)

            if dropout:
                layers[f'dropout{i}'] = nn.Dropout(dropout)
            if pool_size:
                layers[f'maxpool{i}'] = layers_t.QuantizedMaxPool(pool_size, quant_params=quant_params)

            i += 1

        if isinstance(postpool, int) and postpool > 1:
            layers['postpool'] = layers_t.QuantizedAvgPool(postpool, quant_params=quant_params)
        if not isinstance(postpool, int) and math.prod(postpool) > 1:
            layers['postpool'] = layers_t.QuantizedAvgPool(tuple(postpool), quant_params=quant_params)

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
            if quantize_linear:
                layers[f'fc{j}'] = QuantizedLinear(in_units,
                                                   out_units,
                                                   quant_params=quant_params,
                                                   activation=nn.ReLU() if fused_relu else None)
            else:
                layers[f'fc{j}'] = nn.Linear(in_units, out_units)
            if not fused_relu or not quantize_linear:
                layers[f'relu{i}'] = QuantizedReLU(quant_params=quant_params)
            if dropout:
                layers[f'dropout{i}'] = nn.Dropout(dropout)

            i += 1
            j += 1

        if quantize_linear:
            layers[f'fc{j}'] = QuantizedLinear(fc_units[-1] if len(fc_units) > 0 else in_features,
                                               output_shape[0],
                                               quant_params=quant_params)
        else:
            layers[f'fc{j}'] = nn.Linear(fc_units[-1] if len(fc_units) > 0 else in_features, output_shape[0])

        super().__init__(layers)
