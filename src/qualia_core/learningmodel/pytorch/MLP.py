import math
import torch.nn as nn

class MLP(nn.Sequential):
    def __init__(self, input_shape, output_shape, units: tuple=(100, 100, 100)):

        self.input_shape = input_shape
        self.output_shape = output_shape

        layers = [nn.Flatten()]
        if units:
            layers.append(nn.Linear(math.prod(input_shape), units[0]))
            layers.append(nn.ReLU())

            for in_units, out_units in zip(units, units[1:]):
                layers.append(nn.Linear(in_units, out_units))
                layers.append(nn.ReLU())

            layers.append(nn.Linear(units[-1], output_shape[0]))
        else:
            layers.append(nn.Linear(math.prod(input_shape), output_shape[0]))

        super().__init__(*layers)
