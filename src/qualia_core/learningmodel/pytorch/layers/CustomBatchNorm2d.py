import sys

import torch

from .CustomBatchNorm import CustomBatchNorm

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class CustomBatchNorm2d(CustomBatchNorm):
    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Reshape to 1D-like input in order to use CustomBatchNorm forward as-is.

        BatchNorm works on channels with are second dim after batch so flatten the last dimensions to single dim.
        """
        input_shape = input.shape
        input = input.flatten(start_dim=2)

        y = super().forward(input)

        return y.reshape(input_shape)
