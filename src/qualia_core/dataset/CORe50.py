from __future__ import annotations

import logging
import pickle as pkl
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

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

class CORe50(RawDataset):
    """CORe50 object recognition."""

    @dataclass
    class Info:
        path: np.int32
        session: np.int8

    Info_dtype: Final[list[tuple[str, str]]] = [('path', 'U32'), ('session', 'U8')]

    test_list: Final[list[str]] = ['s3', 's7', 's10']

    class_list: Final[dict[int, int]] = {
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 1,
        7: 1,
        8: 1,
        9: 1,
        10: 1,
        11: 2,
        12: 2,
        13: 2,
        14: 2,
        15: 2,
        16: 3,
        17: 3,
        18: 3,
        19: 3,
        20: 3,
        21: 4,
        22: 4,
        23: 4,
        24: 4,
        25: 4,
        26: 5,
        27: 5,
        28: 5,
        29: 5,
        30: 5,
        31: 6,
        32: 6,
        33: 6,
        34: 6,
        35: 6,
        36: 7,
        37: 7,
        38: 7,
        39: 7,
        40: 7,
        41: 8,
        42: 8,
        43: 8,
        44: 8,
        45: 8,
        46: 9,
        47: 9,
        48: 9,
        49: 9,
        50: 9}

    def __init__(self,
                 path: str,
                 variant: str,
                 sessions: list[str] | None = None) -> None:
        super().__init__()
        self.__path = Path(path)
        self.__variant = variant
        self.__sessions = sessions
        self.sets.remove('valid')

    def __load(self, path: Path) -> RawDataModel:
        start = time.time()

        data = np.load(path/'core50_imgs.npz')['x']

        with (path/'paths.pkl').open('rb') as f:
            paths = pkl.load(f)

        train_x_list: list[numpy.typing.NDArray[np.float32]] = []
        train_y_list: list[int] = []
        train_info_list: list[CORe50.Info] = []
        test_x_list: list[numpy.typing.NDArray[np.float32]] = []
        test_y_list: list[int] = []
        test_info_list: list[CORe50.Info] = []
        for x, p in zip(data, paths):
            session, obj, _  = p.split('/')
            label = int(obj[1:])
            if self.__variant == 'category':
                label = self.class_list[label]
            if session in self.test_list:
                test_x_list.append(x.astype(np.float32))
                test_y_list.append(label)
                test_info_list.append(CORe50.Info(path=p, session=session))
            elif self.__sessions is None or session in self.__sessions:
                train_x_list.append(x.astype(np.float32))
                train_y_list.append(label)
                train_info_list.append(CORe50.Info(path=p, session=session))

        train_x = np.array(train_x_list)
        train_y = np.array(train_y_list)
        train_info = np.array(train_info_list, dtype=self.Info_dtype)
        test_x = np.array(test_x_list)
        test_y = np.array(test_y_list)
        test_info = np.array(test_info_list, dtype=self.Info_dtype)

        logger.info('Shapes: train_x=%s, train_y=%s, test_x=%y, test_y=%y',
                    train_x.shape,
                    train_y.shape,
                    test_x.shape,
                    test_y.shape)

        train = RawData(train_x, train_y, train_info)
        test = RawData(test_x, test_y, test_info)


        logger.info('Elapsed: %s s', time.time() - start)

        return RawDataModel(sets=RawDataSets(train=train, test=test), name=self.name)

    @override
    def __call__(self) -> RawDataModel:
        if self.__variant not in ['object', 'category']:
            raise ValueError('Only object or category variants are supported')
        return self.__load(self.__path)

    @property
    @override
    def name(self) -> str:
        return f'{self.__class__.__name__}_{self.__variant}'
