from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing
from qualia_core.datamodel import RawDataModel
from qualia_core.datamodel.RawDataModel import RawData, RawDataSets

from .RawDataset import RawDataset

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)

@dataclass
class CIFAR10File:
    data: numpy.typing.NDArray[np.uint8]
    labels: list[int]
    batch_label: bytes
    filenames: list[bytes]

class CIFAR10(RawDataset):
    def __init__(self, path: str='', dtype: str = 'float32') -> None:
        super().__init__()
        self.__path = Path(path)
        self.__dtype = dtype
        self.sets.remove('valid')

    def __unpickle(self, file: Path) -> CIFAR10File:
        import pickle
        with file.open('rb') as fo:
            raw = pickle.load(fo, encoding='bytes')
            content = {k.decode('cp437'): v for k, v in raw.items()}
            return CIFAR10File(**content)

    def __load_train(self, path: Path) -> RawData:
        start = time.time()

        train_x_list: list[numpy.typing.NDArray[np.uint8]] = []
        train_y_list: list[numpy.typing.NDArray[np.int64]] = []
        for i in range(1, 6):
            d = self.__unpickle(path/f'data_batch_{i}')
            train_x_list.append(d.data)
            train_y_list.append(np.array(d.labels))

        train_x_uint8 = np.concatenate(train_x_list)
        train_x_uint8 = train_x_uint8.reshape((train_x_uint8.shape[0], 3, 32, 32)) # N, C, H, W
        train_x_uint8 = train_x_uint8.transpose((0, 2, 3, 1))
        train_x = train_x_uint8.astype(self.__dtype) # N, H, W, C
        train_y = np.concatenate(train_y_list)

        logger.info('__load_train() Elapsed: %s s', time.time() - start)

        return RawData(train_x, train_y)

    def __load_test(self, path: Path) -> RawData:
        start = time.time()

        d = self.__unpickle(path/'test_batch')

        test_x_uint8 = d.data
        test_x_uint8 = test_x_uint8.reshape((test_x_uint8.shape[0], 3, 32, 32)) # N, C, H, W
        test_x_uint8 = test_x_uint8.transpose((0, 2, 3, 1))
        test_x = test_x_uint8.astype(self.__dtype) # N, H, W, C
        test_y = np.array(d.labels)

        logger.info('__load_test() Elapsed: %s s', time.time() - start)

        return RawData(test_x, test_y)

    @override
    def __call__(self) -> RawDataModel:
        return RawDataModel(sets=RawDataSets(
                                    train=self.__load_train(self.__path),
                                    test=self.__load_test(self.__path),
                                ), name=self.name)
