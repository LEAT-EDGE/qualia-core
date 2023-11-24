from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import numpy.typing

from qualia_core.datamodel import RawDataModel
from qualia_core.datamodel.Info import Info, Info_dtype
from qualia_core.datamodel.RawDataModel import RawData, RawDataSets
from qualia_core.utils.file import DirectoryReader

from .RawDataset import RawDataset

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)

class HD(RawDataset):
    """Heidelberg Digits, raw audio ('hd_audio.tar.gz')."""

    def __init__(self,
                 path: str,
                 variant: str | None = None,
                 test_subjects: list[int] | None = None) -> None:
        super().__init__()
        self.__path = Path(path)
        self.__variant = variant
        self.__test_subjects = test_subjects
        self.__dr = DirectoryReader()
        self.sets.remove('valid')

    def __load_wave(self, recording: Path) -> numpy.typing.NDArray[np.float32]:
        import soundfile as sf
        data, _ = sf.read(str(recording))
        return data.astype(np.float32)

    def __is_test_subject(self, recording: Path, test_list: list[int] | list[str]) -> bool:
        if self.__variant == 'by-subject':
            return int(recording.stem.split('_')[1].replace('speaker-', '')) in test_list
        return recording.name in test_list

    def __load(self, path: Path, test_list: list[int] | list[str]) -> RawDataModel:
        start = time.time()

        directory = self.__dr.read(path / 'audio', ext='.flac', recursive=True)


        train_x_list: list[numpy.typing.NDArray[np.float32]] = []
        train_y_list: list[int] = []
        train_info_list: list[Info] = []
        test_x_list: list[numpy.typing.NDArray[np.float32]] = []
        test_y_list: list[int] = []
        test_info_list: list[Info] = []
        for recording in directory:
            data = self.__load_wave(recording).astype(np.float32)
            label = int(recording.stem.split('-')[-1])
            if self.__is_test_subject(recording, test_list):
                test_x_list.append(data)
                test_y_list.append(label)
                test_info_list.append(Info(subject=int(recording.stem.split('_')[1].replace('speaker-', ''))))
            else:
                train_x_list.append(data)
                train_y_list.append(label)
                train_info_list.append(Info(subject=int(recording.stem.split('_')[1].replace('speaker-', ''))))

        # Zero pad to longest sample
        max_length = max(x.shape[0] for x in train_x_list + test_x_list)
        train_x_list = [np.pad(x, ((0, max_length - x.shape[0])), 'constant', constant_values=0) for x in train_x_list]
        test_x_list = [np.pad(x, ((0, max_length - x.shape[0])), 'constant', constant_values=0) for x in test_x_list]

        train_x = np.expand_dims(np.array(train_x_list), -1) # Add channels dimension
        train_y = np.array(train_y_list)
        train_info = np.array(train_info_list, dtype=Info_dtype)
        test_x = np.expand_dims(np.array(test_x_list), -1) # Add channels dimension
        test_y = np.array(test_y_list)
        test_info = np.array(test_info_list, dtype=Info_dtype)

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
        test_list: list[int] | list[str] = []
        if self.__variant == 'by-subject':
            if self.__test_subjects is None:
                logger.error('test_subects required for by-subject variant')
                raise ValueError
            test_list = self.__test_subjects
        else:
            with (self.__path/'test_filenames.txt').open() as f:
                test_list = f.read().splitlines()


        return self.__load(self.__path, test_list=test_list)

    @property
    @override
    def name(self) -> str:
        if self.__variant:
            return f'{super().name}_{self.__variant}'
        return super().name
