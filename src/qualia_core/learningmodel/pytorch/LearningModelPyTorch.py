import sys

import torch
from torch import nn

from qualia_core.learningmodel.LearningModel import LearningModel

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override


class LearningModelPyTorch(LearningModel, nn.Module):
    def __init__(self, input_shape: tuple[int, ...], output_shape: tuple[int, ...]) -> None:
        self.call_super_init = True # Support multiple inheritance from nn.Module
        super().__init__(input_shape=input_shape, output_shape=output_shape)

    @override
    def __call__(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        return super().__call__(input)
