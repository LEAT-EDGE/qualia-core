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

from qualia_core.datamodel import RawDataModel
from qualia_core.datamodel.RawDataModel import RawData, RawDataSets

from .RawDataset import RawDataset

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


class CIFAR(RawDataset, ABC):
    def __init__(self,  # noqa: PLR0913
                 path: str,
                 dtype: str,
                 labels_field: str,
                 train_files: list[str],
                 test_files: list[str],
                 file_cls: type[CIFARFile]) -> None:
        super().__init__()
        self.__path = Path(path)
        self.__dtype = dtype
        self.__labels_field = labels_field
        self.__train_files = train_files
        self.__test_files = test_files
        self.__file_cls = file_cls
        self.sets.remove('valid')

    def __load_file(self, file: Path) -> CIFARFile:
        with file.open('rb') as fo:
            raw = pickle.load(fo, encoding='bytes')
            content = {k.decode('cp437'): v for k, v in raw.items()}
            return self.__file_cls(**content)

    def __load_batch(self, path: Path) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        d = self.__load_file(path)

        x_uint8 = d.data.reshape((d.data.shape[0], 3, 32, 32))  # N, C, H, W
        x_uint8 = x_uint8.transpose((0, 2, 3, 1))
        x = x_uint8.astype(self.__dtype)  # N, H, W, C
        y = np.array(getattr(d, self.__labels_field))

        return x, y

    def __load_batches(self, files: list[str]) -> RawData:
        start = time.time()

        batches = [self.__load_batch(self.__path / file) for file in files]

        train_x = np.concatenate([b[0] for b in batches])
        train_y = np.concatenate([b[1] for b in batches])

        logger.info('_load_batches() Elapsed: %s s', time.time() - start)

        return RawData(train_x, train_y)

    @override
    def __call__(self) -> RawDataModel:
        return RawDataModel(sets=RawDataSets(train=self.__load_batches(self.__train_files),
                                             test=self.__load_batches(self.__test_files)),
                            name=self.name)
