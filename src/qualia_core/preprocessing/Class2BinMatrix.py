from __future__ import annotations

import logging
import sys

import numpy as np

from qualia_core.datamodel import RawDataModel

from .Preprocessing import Preprocessing

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)

class Class2BinMatrix(Preprocessing[RawDataModel, RawDataModel]):
    """Warning: must be applied after Window."""

    def __init__(self, classes: int | None = None) -> None:
        super().__init__()
        self.__classes = classes

    @override
    def __call__(self, datamodel: RawDataModel) -> RawDataModel:
        for _, s in datamodel:
            if len(s.y.shape) != 1:
                logger.error('Unsupported dimensions: %d, expected 1', len(s.y.shape))
                raise ValueError
            if len(s.y) <= 0: # Handle empty sets
                continue

            if not self.__classes:
                s.y = np.eye(np.max(s.y) + 1, dtype=np.float32)[s.y]
            else:
                s.y = np.eye(self.__classes, dtype=np.float32)[s.y]

        return datamodel
