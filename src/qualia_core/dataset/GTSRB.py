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
from qualia_core.utils.file import CSVReader, DirectoryReader

from .RawDataset import RawDataset

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)

class GTSRB(RawDataset):
    @dataclass
    class Row:
        Filename: str
        Width: str
        Height: str
        Roi_X1: str
        Roi_Y1: str
        Roi_X2: str
        Roi_Y2: str
        ClassId: str

    def __init__(self,
                 path: str='',
                 width: int=32,
                 height: int=32) -> None:
        super().__init__()
        self.__dr = DirectoryReader()
        self.__csvreader: CSVReader[GTSRB.Row] = CSVReader()
        self.__path = Path(path)
        self.__width = width
        self.__height = height
        self.sets.remove('valid')

    def __load_train(self, path: Path) -> RawData:
        import imageio
        from skimage import transform
        start = time.time()
        train_dir = self.__dr.read(path/'Final_Training'/'Images', ext='.ppm', recursive=True)

        train_x_list: list[numpy.typing.NDArray[np.float32]] = []
        train_y_list: list[int] = []
        for image in train_dir:
            imdata = imageio.imread(image)
            imdata = transform.resize(imdata, (self.__width, self.__height))
            imdata = imdata.astype(np.float32)
            train_x_list.append(imdata)
            train_y_list.append(int(image.parent.name))
        train_x = np.array(train_x_list)
        train_y = np.array(train_y_list)

        logger.info('__load_train(): Elapsed: %s s', time.time() - start)

        return RawData(train_x, train_y)

    def __load_test(self, path: Path) -> RawData:
        import imageio
        from skimage import transform
        start = time.time()
        test_dir = path/'Final_Test'/'Images'
        gt = self.__csvreader.read(path/'GT-final_test.csv',
                                   delimiter=';',
                                   labels=GTSRB.Row,
                                   skip_header=True)
        test_x_list: list[numpy.typing.NDArray[np.float32]] = []
        test_y_list: list[int] = []
        for img in gt.content:
            imdata = imageio.imread(test_dir/(img.Filename))
            imdata = transform.resize(imdata, (self.__width, self.__height))
            imdata = imdata.astype(np.float32)
            test_x_list.append(imdata)
            test_y_list.append(int(img.ClassId))
        test_x = np.array(test_x_list)
        test_y = np.array(test_y_list)

        logger.info('__load_test(): Elapsed: %s s', time.time() - start)

        return RawData(test_x, test_y)

    @override
    def __call__(self) -> RawDataModel:
        return RawDataModel(sets=RawDataSets(train=self.__load_train(self.__path), test=self.__load_test(self.__path)),
                            name=self.name)
