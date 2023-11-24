import math
import torch.nn as nn
from qualia_core.learningmodel.pytorch.layers.quantized_layers import QuantizedLinear

class QuantizedMLP(nn.Sequential):
    def __init__(self,
        input_shape,
        output_shape,
        units: tuple=(100, 100, 100),
        bits: int=0,
        quantize_bias: bool=True,
        force_q: int=None):

        bits = int(bits) # Conver TOML integer to Python integer
        print('force_q', force_q, 'bits', bits)

        self.input_shape = input_shape
        self.output_shape = output_shape

        layers = [nn.Flatten(),
                  QuantizedLinear(math.prod(input_shape), units[0], quantize_bias=quantize_bias, bits=bits, force_q=force_q),
                  nn.ReLU()]

        for in_units, out_units in zip(units, units[1:]):
            layers.append(QuantizedLinear(in_units, out_units, quantize_bias=quantize_bias, bits=bits, force_q=force_q))
            layers.append(nn.ReLU())

        layers.append(QuantizedLinear(units[-1], output_shape[0], quantize_bias=quantize_bias, bits=bits, force_q=force_q))

        super().__init__(*layers)
