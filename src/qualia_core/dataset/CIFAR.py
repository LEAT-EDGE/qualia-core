from __future__ import annotations

import logging
import pickle
import sys
import time
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from qualia_core.datamodel.RawDataModel import (
    RawData,
    RawDataChunks,
    RawDataChunksModel,
    RawDataChunksSets,
    RawDataDType,
    RawDataShape,
)
from qualia_core.typing import TYPE_CHECKING

from .RawDataset import RawDatasetChunks

if TYPE_CHECKING:
    from collections.abc import Generator

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)


@dataclass
class CIFARFile:
    data: np.ndarray[Any, np.dtype[np.uint8]]
    batch_label: bytes
    filenames: list[bytes]


class CIFAR(RawDatasetChunks, ABC):
    def __init__(self,  # noqa: PLR0913
                 path: str,
                 dtype: str,
                 labels_field: str,
                 train_files: list[str],
                 test_files: list[str],
                 train_shapes: RawDataShape,
                 test_shapes: RawDataShape,
                 file_cls: type[CIFARFile]) -> None:
        super().__init__()
        self.__path = Path(path)
        self.__dtype = dtype
        self.__labels_field = labels_field
        self.__train_files = train_files
        self.__test_files = test_files
        self.__train_shapes = train_shapes
        self.__test_shapes = test_shapes
        self.__dtypes = RawDataDType(x=np.dtype(dtype), y=np.dtype(np.int64))
        self.__file_cls = file_cls
        self.sets.remove('valid')

    def __load_file(self, file: Path) -> CIFARFile:
        with file.open('rb') as fo:
            raw = pickle.load(fo, encoding='bytes')
            content = {k.decode('cp437'): v for k, v in raw.items()}
            return self.__file_cls(**content)

    def __load_batch(self, path: Path) -> RawData:
        d = self.__load_file(path)

        x_uint8 = d.data.reshape((d.data.shape[0], 3, 32, 32))  # N, C, H, W
        x_uint8 = x_uint8.transpose((0, 2, 3, 1))
        x = x_uint8.astype(self.__dtype)  # N, H, W, C
        y = np.array(getattr(d, self.__labels_field))

        return RawData(x, y)

    def __load_batches(self, files: list[str]) -> Generator[RawData]:
        for file in files:
            start = time.time()
            batch = self.__load_batch(self.__path / file)
            logger.info('"%s" loaded in %s s', self.__path / file, time.time() - start)
            yield batch

    def __load_set(self, files: list[str], shapes: RawDataShape) -> RawDataChunks:
        return RawDataChunks(chunks=self.__load_batches(files),
                             shapes=shapes,
                             dtypes=self.__dtypes)

    @override
    def __call__(self) -> RawDataChunksModel:
        train = self.__load_set(files=self.__train_files, shapes=self.__train_shapes)
        test = self.__load_set(files=self.__test_files, shapes=self.__test_shapes)

        return RawDataChunksModel(sets=RawDataChunksSets(train=train, test=test),
                            name=self.name)
