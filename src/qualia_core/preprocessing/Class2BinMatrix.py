from __future__ import annotations

import logging
import sys

import numpy as np

from qualia_core.datamodel.RawDataModel import RawData, RawDataChunks, RawDataModel

from .Preprocessing import Preprocessing, iterate_generator

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

    @iterate_generator
    def __process_set(self, s: RawData) -> RawData:
        if len(s.y.shape) != 1:
            logger.error('Unsupported dimensions: %d, expected 1', len(s.y.shape))
            raise ValueError
        if len(s.y) <= 0:  # Handle empty sets
            return s

        if not self.__classes:
            s.y = np.eye(np.max(s.y) + 1, dtype=np.float32)[s.y]
        else:
            s.y = np.eye(self.__classes, dtype=np.float32)[s.y]

        return s

    @override
    def __call__(self, datamodel: RawDataModel) -> RawDataModel:
        for sname, s in datamodel:
            # Update the shape and dtype of RawDataChunks instance to get correct pre-allocation of output file
            if isinstance(s, RawDataChunks):
                if not self.__classes:
                    logger.error('classes param is required when using a dataset processed in chunks')
                    raise ValueError
                # Adjust the shape of labels with the one-hot dimension
                s.shapes.labels = (*s.shapes.labels, self.__classes)
                s.dtypes.labels = np.dtype(np.float32)

            setattr(datamodel.sets, sname, self.__process_set(s))

        return datamodel
