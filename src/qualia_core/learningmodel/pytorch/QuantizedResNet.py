from collections import OrderedDict
from typing import cast

from qualia_core.learningmodel.pytorch.layers import QuantizedAdd
from qualia_core.learningmodel.pytorch.layers.quantized_layers import QuantizedIdentity, QuantizedLinear, QuantizedReLU
from qualia_core.learningmodel.pytorch.ResNet import BasicBlock, ResNet
from torch import nn


class QuantizedBasicBlock(BasicBlock):
    def __init__(self,
                    layers_t,
                    in_planes,
                    planes,
                    kernel_size,
                    stride,
                    padding,
                    batch_norm,
                    bn_momentum,
                    force_projection_with_stride,
                    quantize_add,
                    quant_params,
                    fused_relu=True):
        super().__init__(layers_t, in_planes, planes, kernel_size, stride, padding, batch_norm, bn_momentum, force_projection_with_stride)
        self.fused_relu = fused_relu

        
        
        
        
        self.conv1 = layers_t.QuantizedConv(in_planes,
                                            planes,
                                            kernel_size=kernel_size,
                                            stride=1,
                                            padding=padding,
                                            bias=not batch_norm,
                                            quant_params=quant_params,
                                            activation=nn.ReLU() if fused_relu and self.stride == 1 and not batch_norm else None)
        if batch_norm:
            self.bn1 = layers_t.QuantizedBatchNorm(planes,
                                                   momentum=bn_momentum,
                                                   quant_params=quant_params,
                                                   activation=nn.ReLU() if fused_relu and self.stride == 1 else None)


        if self.stride != 1:
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


        if self.stride != 1:
            self.smax = layers_t.QuantizedMaxPool(stride, quant_params=quant_params)
        if self.in_planes != self.expansion*self.planes or force_projection_with_stride and self.stride != 1:
            self.sconv = layers_t.QuantizedConv(in_planes, self.expansion*planes, kernel_size=1, stride=1, bias=not batch_norm, quant_params=quant_params)
            if batch_norm:
                self.sbn = layers_t.QuantizedBatchNorm(
                        self.expansion*planes, momentum=bn_momentum, quant_params=quant_params)

        self.add = QuantizedAdd(quant_params=quant_params , activation=nn.ReLU() if fused_relu else None)
        if fused_relu:
            del self.relu
        else:
            self.relu = QuantizedReLU(quant_params=quant_params)
            
    def forward(self, x):

        out = self.conv1(x)

        if self.batch_norm:
            out = self.bn1(out)
        if self.stride != 1:
            out = self.pool1(out)
        if not self.fused_relu:
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
        if not self.fused_relu:
            out = self.relu(out)
        return out

class QuantizedResNet(ResNet):
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
        
        quantize_linear: bool=True,
        quantize_add: bool=True,
        fused_relu: bool=True,
        quant_params: dict = None):

        #if batch_norm:
        #    raise ValueError('BatchNorm unsupported in quantized Resnet')

        #if prepool > 1:
        #    raise ValueError('AvgPool unsupported in quantized Resnet')
        
        if "bits" in quant_params :
            quant_params["bits"] = int(quant_params["bits"]) # Force conversion from TOML int to plain Python int
            if quant_params["bits"] < 1:
                raise ValueError('bits must be set to a strictly positive integer')
        else :
            raise ValueError('bits must exist in quant_params conf')

        if dims == 1:
            import qualia_core.learningmodel.pytorch.layers1d as layers_t
        elif dims == 2:
            import qualia_core.learningmodel.pytorch.layers2d as layers_t
        else:
            raise ValueError('Only dims=1 or dims=2 supported')

        super().__init__(input_shape=input_shape,
                         output_shape=output_shape,
                         filters=filters,
                         kernel_sizes=kernel_sizes,
                         num_blocks=num_blocks,
                         strides=strides,
                         paddings=paddings,
                         prepool=prepool,
                         postpool=postpool,
                         batch_norm=batch_norm,
                         bn_momentum=bn_momentum,
                         force_projection_with_stride=force_projection_with_stride,
                         dims=dims,
                         basicblockbuilder=lambda in_planes, planes, kernel_size, stride, padding: QuantizedBasicBlock(
                                                                                layers_t=layers_t,
                                                                                in_planes=in_planes,
                                                                                planes=planes,
                                                                                kernel_size=kernel_size,
                                                                                stride=stride,
                                                                                padding=padding,
                                                                                batch_norm=batch_norm,
                                                                                bn_momentum=bn_momentum,
                                                                                force_projection_with_stride=force_projection_with_stride,
                                                                                quantize_add=quantize_add,
                                                                                quant_params=quant_params,
                                                                                fused_relu=fused_relu)
                         )
        self.fused_relu = fused_relu
        self.batch_norm = batch_norm

        self.identity1 = QuantizedIdentity(quant_params=quant_params)
        # Make sure identity1 appears before all of parent ResNet modules to keep a more sensible iterate/print order
        # self._modules is assumed to be an OrderedDict which is the case in PyTorch 2.0
        cast(OrderedDict[str, nn.Module], self._modules).move_to_end('identity1', last=False)

        if prepool > 1:
            self.prepool = layers_t.QuantizedAvgPool(prepool, bits=bits, force_q=force_q)

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
        if fused_relu:
            del self.relu1
        else:
            self.relu1 = QuantizedReLU(quant_params=quant_params)

        if postpool == 'avg':
            self.postpool = layers_t.QuantizedAdaptiveAvgPool(1, quant_params=quant_params)
        elif postpool == 'max':
            self.postpool = layers_t.QuantizedMaxPool(tuple(self._fm_dims), quant_params=quant_params)
        
        self.flatten = nn.Flatten()
        # Replace Linear with the quantized variant
        if quantize_linear:
            self.linear = QuantizedLinear(self.in_planes*QuantizedBasicBlock.expansion, output_shape[0], quant_params=quant_params)

    def forward(self, x):

        out = self.identity1(x)
        
        if hasattr(self, 'prepool'):
            out = self.prepool(out)

        out = self.conv1(out)
        if self.batch_norm:
            out = self.bn1(out)
        if not self.fused_relu:
            out = self.relu1(out)

        for i in range(len(self.layers)):
            out = self.layers[i](out)

        if hasattr(self, 'postpool'):
            out = self.postpool(out)

        out = self.flatten(out)
        out = self.linear(out)
        return out
