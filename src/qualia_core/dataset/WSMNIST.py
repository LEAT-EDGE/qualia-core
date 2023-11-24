from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

from qualia_core.datamodel import RawDataModel
from qualia_core.datamodel.RawDataModel import RawData, RawDataSets

from .RawDataset import RawDataset

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)

class WSMNIST(RawDataset):
    def __init__(self, path: str, variant: str='spoken') -> None:
        super().__init__()
        self.__path = Path(path)
        self.__variant = variant
        self.sets.remove('valid')

    def __load_spoken(self, path: Path) -> RawDataModel:
        train_x = np.load(path/'data_sp_train.npy').reshape(-1, 39, 13).astype(np.float32)
        train_y = np.load(path/'labels_train.npy')
        test_x = np.load(path/'data_sp_test.npy').reshape(-1, 39, 13).astype(np.float32)
        test_y = np.load(path/'labels_test.npy')

        train = RawData(train_x, train_y)
        test = RawData(test_x, test_y)

        return RawDataModel(sets=RawDataSets(train=train, test=test), name=self.name)

    def __load_written(self, path: Path) -> RawDataModel:
        train_x = np.load(path/'data_wr_train.npy').reshape(-1, 28, 28, 1).astype(np.float32)
        train_y = np.load(path/'labels_train.npy')
        test_x = np.load(path/'data_wr_test.npy').reshape(-1, 28, 28, 1).astype(np.float32)
        test_y = np.load(path/'labels_test.npy')

        train = RawData(train_x, train_y)
        test = RawData(test_x, test_y)

        return RawDataModel(sets=RawDataSets(train=train, test=test), name=self.name)

    @override
    def __call__(self) -> RawDataModel:
        if self.__variant == 'spoken':
            return self.__load_spoken(self.__path)
        if self.__variant == 'written':
            return self.__load_written(self.__path)

        logger.error('Only spoken or written variants supported')
        raise ValueError

    @property
    @override
    def name(self) -> str:
        return f'{super().name}_{self.__variant}'
