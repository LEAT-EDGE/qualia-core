from abc import ABC, abstractmethod

import torch

from qualia_core.dataaugmentation.DataAugmentation import DataAugmentation


class DataAugmentationPyTorch(ABC, DataAugmentation):
    @abstractmethod
    def __call__(self, data: tuple[torch.Tensor, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        ...
