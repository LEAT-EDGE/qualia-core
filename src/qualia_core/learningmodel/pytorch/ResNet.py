import math
import numpy as np


import torch.nn as nn
import torch.nn.functional as F
from qualia_core.learningmodel.pytorch.layers import Add


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                    layers_t,
                    in_planes,
                    planes,
                    kernel_size,
                    stride,
                    padding,
                    batch_norm,
                    bn_momentum,
                    force_projection_with_stride):
        


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
        
        
        self.conv2 = layers_t.Conv(
            planes, planes, kernel_size=kernel_size, stride=1, padding=padding, bias=not batch_norm)
        if batch_norm:
            self.bn2 = layers_t.BatchNorm(planes, momentum=bn_momentum)
        if self.stride != 1:
            self.smax = layers_t.MaxPool(stride)
        if self.in_planes != self.expansion*self.planes or force_projection_with_stride and self.stride != 1:
            self.sconv = layers_t.Conv(in_planes, self.expansion*planes, kernel_size=1, stride=1, bias=not batch_norm)
            if batch_norm:
                self.sbn = layers_t.BatchNorm(
                    self.expansion*planes, momentum=bn_momentum)
        self.add = Add()
        self.relu = nn.ReLU()

    def forward(self, x):

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

        if self.in_planes != self.expansion*self.planes or self.force_projection_with_stride and self.stride != 1:
            tmp = self.sconv(tmp)

            if self.batch_norm:
                tmp = self.sbn(tmp)

        if self.stride != 1:
            tmp = self.smax(tmp)

        out = self.add(out, tmp)
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self,
        input_shape,
        output_shape,
        filters: tuple=(15, 15, 30, 60, 120),
        kernel_sizes: tuple=(3, 3, 3, 3, 3),

        num_blocks: tuple=(2, 2, 2, 2),
        strides: tuple=(1, 1, 2, 2, 2),
        paddings: int=(1, 1, 1, 1, 1),
        prepool: int=1,
        postpool: str='max',
        batch_norm: bool=False,
        bn_momentum: float=0.1,
        force_projection_with_stride: bool=True,

        
        dims: int=1,
        basicblockbuilder=None):
        super().__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape

        if dims == 1:
            import qualia_core.learningmodel.pytorch.layers1d as layers_t
        elif dims == 2:
            import qualia_core.learningmodel.pytorch.layers2d as layers_t
        else:
            raise ValueError('Only dims=1 or dims=2 supported')

        if basicblockbuilder is None:
            basicblockbuilder=lambda in_planes, planes, kernel_size, stride, padding: BasicBlock(
                layers_t=layers_t,
                in_planes=in_planes,
                planes=planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                batch_norm=batch_norm,
                bn_momentum=bn_momentum,
                force_projection_with_stride=force_projection_with_stride)

        self.in_planes = filters[0]
        self.batch_norm = batch_norm
        self.num_blocks = num_blocks
        if prepool > 1:
            self.prepool = layers_t.AvgPool(prepool)

        self.conv1 = layers_t.Conv(input_shape[-1], filters[0], kernel_size=kernel_sizes[0], stride=strides[0], padding=paddings[0], bias=not batch_norm)
        if self.batch_norm:
            self.bn1 = layers_t.BatchNorm(self.in_planes, momentum=bn_momentum)
        self.relu1 = nn.ReLU()

        self.layers = []
        for planes, kernel_size, stride, padding, num_block in zip(filters[1:], kernel_sizes[1:], strides[1:], paddings[1:], num_blocks):
            self.layers += [self._make_layer(basicblockbuilder, num_block, planes, kernel_size, stride, padding)]
        self.layers = nn.ModuleList(self.layers)

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

    def _make_layer(self, basicblockbuilder, num_blocks, planes, kernel_size, stride, padding):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            block = basicblockbuilder(in_planes=self.in_planes, planes=planes, kernel_size=kernel_size, stride=stride, padding=padding)
            layers.append(block)
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
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
        out = self.linear(out)
        return out
