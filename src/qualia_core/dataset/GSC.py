from __future__ import annotations

import logging
import os
import sys
import time
import wave
from concurrent.futures import Future, ProcessPoolExecutor
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any, Callable, Final

import numpy as np
import numpy.typing
from qualia_core.datamodel import RawDataModel
from qualia_core.datamodel.RawDataModel import RawData, RawDataSets
from qualia_core.utils.file import DirectoryReader

from .RawDataset import RawDataset

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)

class GSC(RawDataset):
    class_list_no_background_noise: Final[dict[str, int | None]] = {
        '_background_noise_': None, # Drop background noise
        'backward': 0,
        'bed': 1,
        'bird': 2,
        'cat': 3,
        'dog': 4,
        'down': 5,
        'eight': 6,
        'five': 7,
        'follow': 8,
        'forward': 9,
        'four': 10,
        'go': 11,
        'happy': 12,
        'house': 13,
        'learn': 14,
        'left': 15,
        'marvin': 16,
        'nine': 17,
        'no': 18,
        'off': 19,
        'on': 20,
        'one': 21,
        'right': 22,
        'seven': 23,
        'sheila': 24,
        'six': 25,
        'stop': 26,
        'three': 27,
        'tree': 28,
        'two': 29,
        'up': 30,
        'visual': 31,
        'wow': 32,
        'yes': 33,
        'zero': 34,
    }

    class_list_digits: Final[dict[str, int | None]] = {
        '_background_noise_': None, # Drop background noise
        'backward': None,
        'bed': None,
        'bird': None,
        'cat': None,
        'dog': None,
        'down': None,
        'eight': 8,
        'five': 5,
        'follow': None,
        'forward': None,
        'four': 4,
        'go': None,
        'happy': None,
        'house': None,
        'learn': None,
        'left': None,
        'marvin': None,
        'nine': 9,
        'no': None,
        'off': None,
        'on': None,
        'one': 1,
        'right': None,
        'seven': 7,
        'sheila': None,
        'six': 6,
        'stop': None,
        'three': 3,
        'tree': None,
        'two': 2,
        'up': None,
        'visual': None,
        'wow': None,
        'yes': None,
        'zero': 0,
    }

    def __init__(self, path: str, variant: str = 'v2', subset: str = 'digits', train_valid_split: bool = False) -> None:
        super().__init__()
        self.__path = Path(path)
        self.__variant = variant
        self.__subset = subset
        self.__dr = DirectoryReader()
        self.__train_valid_split = train_valid_split
        if not train_valid_split:
            self.sets.remove('valid')

    def load_wave(self, recording: Path) -> numpy.typing.NDArray[np.float32]:
        with wave.open(str(recording)) as f:
            data = f.readframes(f.getnframes())

        data_array = np.frombuffer(data, dtype=np.int16).copy().astype(np.float32)
        data_array.resize((16000, 1)) # Resize to 1s (16kHz) with zero-padding, 1 channel
        return data_array

    def _load_files(self,
                    i: int,
                    files: numpy.typing.NDArray[Any], # Path or object not supported as NDArray generic
                    loader: Callable[[Path],
                                     numpy.typing.NDArray[np.float32]]) -> tuple[str, tuple[int, ...], numpy.typing.DTypeLike]:
        start = time.time()

        data_list: list[numpy.typing.NDArray[np.float32]] = []
        logger.info('Process %s loading %s files...', i, len(files))

        for file in files:
            data = loader(file)
            data_list.append(data)

        data_array = np.array(data_list)
        del data_list

        data_buffer = SharedMemory(size=data_array.nbytes, create=True)

        data_shared = np.frombuffer(data_buffer.buf, dtype=data_array.dtype).reshape(data_array.shape)

        np.copyto(data_shared, data_array)

        del data_shared

        ret = (data_buffer.name, data_array.shape, data_array.dtype)

        data_buffer.close()

        logger.info('Process %s finished in %s s.', i, time.time() - start)
        return ret

    def __threaded_loader(self,
                          training_files: list[Path],
                          testing_files: list[Path],
                          validation_files: list[Path],
                          loader: Callable[[Path],
                                           numpy.typing.NDArray[np.float32]]) -> tuple[numpy.typing.NDArray[np.float32] | None,
                                                                                       numpy.typing.NDArray[np.float32] | None,
                                                                                       numpy.typing.NDArray[np.float32] | None]:
        cpus: int | None = os.cpu_count()
        total_chunks: int = cpus // 2 if cpus is not None else 2
        total_files = len(training_files) + len(validation_files) + len(testing_files)
        training_chunks = min(len(training_files), max(1, len(training_files) * total_chunks // total_files))
        validation_chunks = min(len(validation_files), max(1, len(validation_files) * total_chunks // total_files))
        testing_chunks = min(len(testing_files), max(1, total_chunks - training_chunks - validation_chunks))
        training_files_chunks = np.array_split(np.array(training_files), training_chunks) if training_files else []
        validation_files_chunks = np.array_split(np.array(validation_files), validation_chunks) if validation_files else []
        testing_files_chunks = np.array_split(np.array(testing_files), testing_chunks) if testing_files else []

        logger.info('Using %s threads for training data, %s threads for validation data and %s threads for testing data',
                    training_chunks,
                    validation_chunks,
                    testing_chunks)

        with SharedMemoryManager() as smm:
            with ProcessPoolExecutor() as executor:
                train_futures = [executor.submit(self._load_files, i, files, loader)
                           for i, files in enumerate(training_files_chunks)]
                valid_futures = [executor.submit(self._load_files, i, files, loader)
                           for i, files in enumerate(validation_files_chunks)]
                test_futures = [executor.submit(self._load_files, i, files, loader)
                           for i, files in enumerate(testing_files_chunks)]

                def load_results(futures: list[Future[tuple[str,
                                                            tuple[int, ...],
                                                            numpy.typing.DTypeLike]]]) -> numpy.typing.NDArray[np.float32] | None:

                    names = [f.result()[0] for f in futures]
                    shapes = [f.result()[1] for f in futures]
                    dtypes = [f.result()[2] for f in futures]
                    bufs = [SharedMemory(n) for n in names]

                    data_list = [np.frombuffer(buf.buf, dtype=dtype).reshape(shape)
                              for shape, dtype, buf in zip(shapes, dtypes, bufs)]

                    data_array = np.concatenate(data_list) if data_list else None
                    del data_list

                    for buf in bufs:
                        buf.unlink()

                    return data_array

                train_x_array = load_results(train_futures)
                valid_x_array = load_results(valid_futures)
                test_x_array = load_results(test_futures)

        return train_x_array, valid_x_array, test_x_array


    def __load_v2(self, path: Path, class_list: dict[str, int | None]) -> RawDataModel:
        start = time.time()

        directory = self.__dr.read(path, ext='.wav', recursive=True)

        with (path/'validation_list.txt').open() as f:
            validation_list = f.read().splitlines()

        with (path/'testing_list.txt').open() as f:
            testing_list = f.read().splitlines()

        # Build files list for train and test
        training_files: list[Path] = []
        validation_files: list[Path] = []
        testing_files: list[Path] = []
        training_labels: list[int] = []
        validation_labels: list[int] = []
        testing_labels: list[int] = []
        for file in list(directory):
            label = class_list[file.parent.name]
            if label is None: # Drop sample excluded from class list
                continue
            if str(file.relative_to(path)) in testing_list:
                testing_files.append(file)
                testing_labels.append(label)
            elif self.__train_valid_split and str(file.relative_to(path)) in validation_list:
                validation_files.append(file)
                validation_labels.append(label)
            else:
                training_files.append(file)
                training_labels.append(label)

        train_x, valid_x, test_x = self.__threaded_loader(training_files, testing_files, validation_files, loader=self.load_wave)

        train_y = np.array(training_labels) if training_labels else None
        valid_y = np.array(validation_labels) if validation_labels else None
        test_y = np.array(testing_labels) if testing_labels else None
        logger.info('Shapes: train_x=%s, train_y=%s, valid_x=%s, valid_y=%s, test_x=%s, test_y=%s',
                    train_x.shape if train_x is not None else None,
                    train_y.shape if train_y is not None else None,
                    valid_x.shape if valid_x is not None else None,
                    valid_y.shape if valid_y is not None else None,
                    test_x.shape if test_x is not None else None,
                    test_y.shape if test_y is not None else None)

        train = RawData(train_x, train_y) if train_x is not None and train_y is not None else None
        valid = RawData(valid_x, valid_y) if valid_x is not None and valid_y is not None else None
        test = RawData(test_x, test_y) if test_x is not None and test_y is not None else None

        logger.info('Elapsed: %s s', time.time() - start)

        return RawDataModel(sets=RawDataSets(train=train, valid=valid, test=test), name=self.name)

    @override
    def __call__(self) -> RawDataModel:
        if self.__variant != 'v2':
            raise ValueError('Only v2 variant supported')

        if self.__subset == 'digits':
            class_list = GSC.class_list_digits
        elif self.__subset == 'no_background_noise':
            class_list = GSC.class_list_no_background_noise
        else:
            raise ValueError('Only digits or no_background_noise subsets supported')

        return self.__load_v2(self.__path, class_list=class_list)

    @property
    @override
    def name(self) -> str:
        return f'{super().name}_{self.__variant}_{self.__subset}'
