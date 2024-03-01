from __future__ import annotations

import logging
import pickle
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

class BrainMIX(RawDataset):
    def __init__(self, path: str) -> None:
        super().__init__()
        self.__path = Path(path)
        self.sets.remove('valid')

    @override
    def __call__(self) -> RawDataModel:
        with (self.__path/'traindata48_shuffled.pickle').open('rb') as fd:
            traindata = pickle.load(fd)
        with (self.__path/'valid48.pickle').open('rb') as fd:
            testdata = pickle.load(fd)

        train_x = traindata['signal']
        train_y = np.expand_dims(traindata['truth'], axis=1).astype(np.float32)
        test_x = testdata['signal']
        test_y = np.expand_dims(testdata['truth'], axis=1).astype(np.float32)

        train = RawData(train_x, train_y)
        test = RawData(test_x, test_y)

        return RawDataModel(sets=RawDataSets(train=train, test=test), name=self.name)
