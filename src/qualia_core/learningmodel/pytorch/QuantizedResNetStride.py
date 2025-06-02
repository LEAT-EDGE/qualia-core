from __future__ import annotations

import logging
import sys
from collections import OrderedDict

import torch
from torch import nn

from qualia_core.learningmodel.pytorch.layers import QuantizedAdd
from qualia_core.learningmodel.pytorch.layers.quantized_layers import QuantizedIdentity, QuantizedLinear, QuantizedReLU
from qualia_core.learningmodel.pytorch.ResNetStride import BasicBlock, ResNetStride
from qualia_core.typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType  # noqa: TCH003

    from qualia_core.learningmodel.pytorch.Quantizer import QuantizationConfig  # noqa: TCH001

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)


class QuantizedBasicBlock(BasicBlock):
    def __init__(self,  # noqa: PLR0913
                 layers_t: ModuleType,
                 in_planes: int,
                 planes: int,
                 kernel_size: int,
                 pool_size: int,
                 stride: int,
                 padding: int,
                 batch_norm: bool,  # noqa: FBT001
                 bn_momentum: float,
                 force_projection_with_pooling: bool,  # noqa: FBT001

                 quant_params: QuantizationConfig,
                 fused_relu: bool = True) -> None:  # noqa: FBT001, FBT002
        super().__init__(layers_t=layers_t,
                         in_planes=in_planes,
                         planes=planes,
                         kernel_size=kernel_size,
                         pool_size=pool_size,
                         stride=stride,
                         padding=padding,
                         batch_norm=batch_norm,
                         bn_momentum=bn_momentum,
                         force_projection_with_pooling=force_projection_with_pooling)

        self.fused_relu = fused_relu

        self.conv1 = layers_t.QuantizedConv(in_planes,
                                            planes,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=padding,
                                            bias=not batch_norm,
                                            quant_params=quant_params,
                                            activation=nn.ReLU() if fused_relu and self.stride == 1 and not batch_norm else None)
        if batch_norm:
            self.bn1 = layers_t.QuantizedBatchNorm(planes,
                                                   momentum=bn_momentum,
                                                   quant_params=quant_params,
                                                   activation=nn.ReLU() if fused_relu and self.stride == 1 else None)

        if self.pool_size != 1:
            self.pool1 = layers_t.QuantizedMaxPool(stride, quant_params=quant_params, activation=nn.ReLU() if fused_relu else None)
        if fused_relu:
            del self.relu1
        else:
            self.relu1 = QuantizedReLU(quant_params=quant_params)

        self.conv2 = layers_t.QuantizedConv(planes,
                                            planes,
                                            kernel_size=kernel_size,
                                            stride=1,
                                            padding=padding,
                                            bias=not batch_norm,
                                            quant_params=quant_params)

        if batch_norm:
            self.bn2 = layers_t.QuantizedBatchNorm(planes, momentum=bn_momentum, quant_params=quant_params)

        if self.pool_size != 1:
            self.spool = layers_t.QuantizedMaxPool(stride, quant_params=quant_params)
        if (self.in_planes != self.expansion * self.planes
            or force_projection_with_pooling and self.pool_size != 1
            or self.stride != 1):
            self.sconv = layers_t.QuantizedConv(in_planes,
                                                self.expansion * planes,
                                                kernel_size=1,
                                                stride=stride,
                                                bias=not batch_norm,
                                                quant_params=quant_params)
            if batch_norm:
                self.sbn = layers_t.QuantizedBatchNorm(
                        self.expansion * planes, momentum=bn_momentum, quant_params=quant_params)

        self.add = QuantizedAdd(quant_params=quant_params, activation=nn.ReLU() if fused_relu else None)
        if fused_relu:
            del self.relu
        else:
            self.relu = QuantizedReLU(quant_params=quant_params)

    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input

        out = self.conv1(x)

        if self.batch_norm:
            out = self.bn1(out)
        if self.pool_size != 1:
            out = self.pool1(out)
        if not self.fused_relu:
            out = self.relu1(out)

        out = self.conv2(out)
        if self.batch_norm:
            out = self.bn2(out)

        # shortcut
        tmp = x

        if (self.in_planes != self.expansion * self.planes
            or self.force_projection_with_pooling and self.pool_size != 1
            or self.stride != 1):
            tmp = self.sconv(tmp)

            if self.batch_norm:
                tmp = self.sbn(tmp)

        if self.pool_size != 1:
            tmp = self.spool(tmp)

        out = self.add(out, tmp)
        if not self.fused_relu:
            out = self.relu(out)
        return out


class QuantizedResNetStride(ResNetStride):
    def __init__(self,  # noqa: PLR0913
                 input_shape: tuple[int, ...],
                 output_shape: tuple[int, ...],
                 filters: list[int],
                 kernel_sizes: list[int],
                 pool_sizes: list[int],

                 num_blocks: list[int],
                 strides: list[int],
                 paddings: list[int],

                 quant_params: QuantizationConfig,

                 prepool: int = 1,
                 postpool: str = 'max',
                 batch_norm: bool = False,  # noqa: FBT001, FBT002
                 bn_momentum: float = 0.1,
                 force_projection_with_pooling: bool = True,  # noqa: FBT001, FBT002

                 dims: int = 1,
                 fused_relu: bool = True) -> None:  # noqa: FBT001, FBT002

        if dims == 1:
            import qualia_core.learningmodel.pytorch.layers1d as layers_t
        elif dims == 2:  # noqa: PLR2004
            import qualia_core.learningmodel.pytorch.layers2d as layers_t
        else:
            logger.error('Only dims=1 or dims=2 supported, got: %s', dims)
            raise ValueError

        super().__init__(input_shape=input_shape,
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
                         force_projection_with_pooling=force_projection_with_pooling,
                         dims=dims,
                         basicblockbuilder=lambda in_planes, planes, kernel_size, pool_size, stride, padding: QuantizedBasicBlock(
                                                                                layers_t=layers_t,
                                                                                in_planes=in_planes,
                                                                                planes=planes,
                                                                                kernel_size=kernel_size,
                                                                                pool_size=pool_size,
                                                                                stride=stride,
                                                                                padding=padding,
                                                                                batch_norm=batch_norm,
                                                                                bn_momentum=bn_momentum,
                                                                                force_projection_with_pooling=force_projection_with_pooling,
                                                                                quant_params=quant_params,
                                                                                fused_relu=fused_relu),
                         )
        self.fused_relu = fused_relu
        self.batch_norm = batch_norm

        self.identity1 = QuantizedIdentity(quant_params=quant_params)
        # Make sure identity1 appears before all of parent ResNet modules to keep a more sensible iterate/print order
        if isinstance(self._modules, OrderedDict):
            # self._modules is and OrderedDict in PyTorch 2.0
            self._modules.move_to_end('identity1', last=False)
        else:
            # self._modules is not an OrderedDict anymore in PyTorch 2.5 so create a new dict with correct order
            self._modules = {'identity1': self._modules.pop('identity1'), **self._modules}

        if prepool > 1:
            self.prepool = layers_t.QuantizedAvgPool(prepool, quant_params=quant_params)

        self.conv1 = layers_t.QuantizedConv(input_shape[-1],
                                            filters[0],
                                            kernel_size=kernel_sizes[0],
                                            stride=strides[0],
                                            padding=paddings[0],
                                            bias=not batch_norm,
                                            quant_params=quant_params,
                                            activation=nn.ReLU() if fused_relu and not batch_norm else None)
        if self.batch_norm:
            self.bn1 = layers_t.QuantizedBatchNorm(self.in_planes,
                                                   momentum=bn_momentum,
                                                   quant_params=quant_params,
                                                   activation=nn.ReLU() if fused_relu else None)
        if pool_sizes[0] != 1:
            self.pool1 = layers_t.QuantizedMaxPool(pool_sizes[0], quant_params=quant_params)
        if fused_relu:
            del self.relu1
        else:
            self.relu1 = QuantizedReLU(quant_params=quant_params)

        if postpool == 'sum':
            self.conv2 = layers_t.QuantizedConv(input_shape[-1],
                                                filters[0],
                                                kernel_size=kernel_sizes[0],
                                                stride=strides[0],
                                                padding=paddings[0],
                                                bias=not batch_norm,
                                                quant_params=quant_params,
                                                activation=nn.ReLU() if fused_relu and not batch_norm else None)
            if fused_relu:
                del self.relu2
            else:
                self.relu2 = QuantizedReLU(quant_params=quant_params)
            self.gsp = layers_t.QuantizedGlobalSumPool(quant_params=quant_params)
        else:
            if postpool == 'avg':
                self.postpool = layers_t.QuantizedAdaptiveAvgPool(1, quant_params=quant_params)
            elif postpool == 'max':
                self.postpool = layers_t.QuantizedMaxPool(tuple(self._fm_dims), quant_params=quant_params)

            self.flatten = nn.Flatten()
            self.linear = QuantizedLinear(self.in_planes * QuantizedBasicBlock.expansion, output_shape[0], quant_params=quant_params)

    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input

        out = self.identity1(x)

        if hasattr(self, 'prepool'):
            out = self.prepool(out)

        out = self.conv1(out)
        if self.batch_norm:
            out = self.bn1(out)
        if hasattr(self, 'pool1'):
            out = self.pool1(out)
        if not self.fused_relu:
            out = self.relu1(out)

        for i in range(len(self.layers)):
            out = self.layers[i](out)

        if hasattr(self, 'gsp'):
            out = self.conv2(out)
            out = self.relu2(out)
            out = self.gsp(out)
        else:
            if hasattr(self, 'postpool'):
                out = self.postpool(out)

            out = self.flatten(out)
            out = self.linear(out)

        return out
