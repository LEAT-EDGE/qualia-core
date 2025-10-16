from __future__ import annotations

import logging
from dataclasses import dataclass

from .CIFAR import CIFAR, CIFARFile

logger = logging.getLogger(__name__)


@dataclass
class CIFAR10File(CIFARFile):
    labels: list[int]


class CIFAR10(CIFAR):
    def __init__(self,
                 path: str,
                 dtype: str = 'float32') -> None:
        super().__init__(path=path,
                         dtype=dtype,
                         labels_field='labels',
                         train_files=[f'data_batch_{i}' for i in range(1, 6)],
                         test_files=['test_batch'],
                         file_cls=CIFAR10File)
