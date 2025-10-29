from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from qualia_core.datamodel.RawDataModel import RawDataShape

from .CIFAR10 import CIFAR, CIFARFile

logger = logging.getLogger(__name__)


@dataclass
class CIFAR100File(CIFARFile):
    coarse_labels: list[int]
    fine_labels: list[int]


class CIFAR100(CIFAR):
    def __init__(self,
                 path: str,
                 dtype: str = 'float32',
                 labels_field: Literal['coarse_labels', 'fine_labels'] = 'fine_labels') -> None:
        super().__init__(path=path,
                         dtype=dtype,
                         labels_field=labels_field,
                         train_files=['train'],
                         test_files=['test'],
                         train_shapes=RawDataShape(x=(None, 32, 32, 3), y=(None,)),
                         test_shapes=RawDataShape(x=(None, 32, 32, 3), y=(None,)),
                         file_cls=CIFAR100File)
