from __future__ import annotations

import logging
import math
import os
import sys
import time
import wave
from concurrent.futures import Future, ProcessPoolExecutor
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any, Callable, Final

import numpy as np
import numpy.typing

from qualia_core.datamodel import RawDataModel
from qualia_core.datamodel.RawDataModel import RawData, RawDataSets
from qualia_core.utils.file import DirectoryReader
from qualia_core.utils.process.init_process import init_process
from qualia_core.utils.process.SharedMemoryManager import SharedMemoryManager

from .RawDataset import RawDataset

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)

class GSC(RawDataset):
    # 35 classes
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

    # 10 classes
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

    # 12 classes
    # https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/datasets/speech_commands/speech_commands_dataset_builder.py
    class_list_tensorflow: Final[dict[str, int | None]] = {
        '_background_noise_': 10,  # silence
        '_silence_': 10,
        '_unknown_': 11,
        'backward': 11,  # unknown
        'bed': 11,  # unknown
        'bird': 11,  # unknown
        'cat': 11,  # unknown
        'dog': 11,  # unknown
        'down': 0,
        'eight': 11,  # unknown
        'five': 11,  # unknown
        'follow': 11,  # unknown
        'forward': 11,  # unknown
        'four': 11,  # unknown
        'go': 1,
        'happy': 11,  # unknown
        'house': 11,  # unknown
        'learn': 11,  # unknown
        'left': 2,
        'marvin': 11,  # unknown
        'nine': 11,  # unknown
        'no': 3,
        'off': 4,
        'on': 5,
        'one': 11,  # unknown
        'right': 6,
        'seven': 11,  # unknown
        'sheila': 11,  # unknown
        'six': 11,  # unknown
        'stop': 7,
        'three': 11,  # unknown
        'tree': 11,  # unknown
        'two': 11,  # unknown
        'up': 8,
        'visual': 11,  # unknown
        'wow': 11,  # unknown
        'yes': 9,
        'zero': 11,  # unknown
    }

    def __init__(self,  # noqa: PLR0913, PLR0917
                 path: str,
                 test_path: str | None = None,
                 variant: str = 'v2',
                 subset: str = 'digits',
                 train_valid_split: bool = False,  # noqa: FBT001, FBT002
                 record_length: int = 16000) -> None:
        super().__init__()
        self._path = Path(path)
        self._test_path = Path(test_path) if test_path is not None else None
        self.__variant = variant
        self.__subset = subset
        self.__dr = DirectoryReader()
        self.__train_valid_split = train_valid_split
        if not train_valid_split:
            self.sets.remove('valid')
        self.__record_length = record_length

    def load_wave(self, recording: Path) -> np.ndarray[Any, np.dtype[np.float32]]:
        with wave.open(str(recording)) as f:
            data = f.readframes(f.getnframes())

        return np.frombuffer(data, dtype=np.int16).copy().astype(np.float32)

    def load_and_resize_wave(self, recording: Path) -> np.ndarray[Any, np.dtype[np.float32]]:
        data_array = self.load_wave(recording)
        data_array.resize((self.__record_length, 1))  # Resize to 1s (16kHz) with zero-padding, 1 channel
        return data_array

    def load_and_split_wave(self, recording: Path) -> np.ndarray[Any, np.dtype[np.float32]]:
        data_array = self.load_wave(recording)
        stride = self.__record_length // 2
        shape = (data_array.size - self.__record_length + 1, self.__record_length)
        strides = data_array.strides * 2
        view = np.lib.stride_tricks.as_strided(data_array, strides=strides, shape=shape)[0::stride]
        return np.expand_dims(view.copy(), -1)

    def _load_files(self,
                    smm_address: str | tuple[str, int],
                    i: int,
                    files: np.ndarray[Any, Any], # Path or object not supported as NDArray generic
                    loader: Callable[[Path], np.ndarray[Any, np.dtype[np.float32]]]) -> tuple[str,
                                                                                              tuple[int, ...],
                                                                                              numpy.typing.DTypeLike]:
        start = time.time()

        smm = SharedMemoryManager(address=smm_address)
        smm.connect()

        data_list: list[np.ndarray[Any, np.dtype[np.float32]]] = []
        logger.info('Process %s loading %s files...', i, len(files))

        for file in files:
            data = loader(file)
            data_list.append(data)

        data_array = np.array(data_list)
        del data_list

        data_buffer = smm.SharedMemory(size=data_array.nbytes)

        data_shared = np.frombuffer(data_buffer.buf, count=data_array.size, dtype=data_array.dtype).reshape(data_array.shape)

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
                          loader: Callable[[Path], np.ndarray[Any, np.dtype[np.float32]]]) -> tuple[
                                  np.ndarray[Any, np.dtype[np.float32]] | None,
                                  np.ndarray[Any, np.dtype[np.float32]] | None,
                                  np.ndarray[Any, np.dtype[np.float32]] | None]:
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

        with SharedMemoryManager() as smm, ProcessPoolExecutor(initializer=init_process) as executor:
            if smm.address is None: # After smm is started in context, address is necessary non-None
                raise RuntimeError

            train_futures = [executor.submit(self._load_files, smm.address, i, files, loader)
                       for i, files in enumerate(training_files_chunks)]
            valid_futures = [executor.submit(self._load_files, smm.address, i, files, loader)
                       for i, files in enumerate(validation_files_chunks)]
            test_futures = [executor.submit(self._load_files, smm.address, i, files, loader)
                       for i, files in enumerate(testing_files_chunks)]

            def load_results(futures: list[Future[tuple[str,
                                                        tuple[int, ...],
                                                        numpy.typing.DTypeLike]]]) -> np.ndarray[Any, np.dtype[np.float32]] | None:

                names = [f.result()[0] for f in futures]
                shapes = [f.result()[1] for f in futures]
                dtypes = [f.result()[2] for f in futures]
                bufs = [SharedMemory(n) for n in names]

                data_list = [np.frombuffer(buf.buf, count=math.prod(shape), dtype=dtype).reshape(shape)
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

    def _load_v2(self, path: Path, class_list: dict[str, int | None]) -> RawDataModel:
        start = time.time()

        directory = self.__dr.read(path, ext='.wav', recursive=True)

        with (path/'validation_list.txt').open() as f:
            validation_list = f.read().splitlines()

        # For tensorflow 12-class subset, these files are excluded from train set but the test set is built from the train archive
        with (path/'testing_list.txt').open() as f:
            testing_list = f.read().splitlines()

        # Build files list for train and test
        training_files: list[Path] = []
        validation_files: list[Path] = []
        testing_files: list[Path] = []
        bg_noise_training_files: list[Path] = []
        training_labels: list[int] = []
        validation_labels: list[int] = []
        testing_labels: list[int] = []
        for file in list(directory):
            label = class_list[file.parent.name]
            if label is None:  # Drop sample excluded from class list
                continue
            if file.parent.name == '_background_noise_':  # Special handling needed for background noise
                if file.name != 'running_tap.wav':  # This specific file is used as validation, exclude from training
                    bg_noise_training_files.append(file)
            elif file.relative_to(path).as_posix() in testing_list:
                testing_files.append(file)
                testing_labels.append(label)
            elif self.__train_valid_split and file.relative_to(path).as_posix() in validation_list:
                validation_files.append(file)
                validation_labels.append(label)
            else:
                training_files.append(file)
                training_labels.append(label)

        # tensorflow-dataset 12 classes subset uses a separate test archive
        if self.__subset == 'tensorflow':
            if self._test_path is None:
                logger.error('Missing params.test_path, required for tensorflow subset')
                raise ValueError
            test_directory = self.__dr.read(self._test_path, ext='.wav', recursive=True)
            for file in list(test_directory):
                label = class_list[file.parent.name]
                if label is None:
                    continue
                testing_files.append(file)
                testing_labels.append(label)

        train_x, valid_x, test_x = self.__threaded_loader(training_files,
                                                          testing_files,
                                                          validation_files,
                                                          loader=self.load_and_resize_wave)

        if class_list['_background_noise_'] is not None:
            logger.info('Loading background noise...')
            bg_noise_train_x = np.concatenate([self.load_and_split_wave(file) for file in bg_noise_training_files])
            bg_noise_valid_x = self.load_and_split_wave(path/'_background_noise_'/'running_tap.wav')

            train_x = np.concatenate((train_x, bg_noise_train_x)) if train_x is not None else bg_noise_train_x
            training_labels += [class_list['_background_noise_']] * len(bg_noise_train_x)

            valid_x = np.concatenate((valid_x, bg_noise_valid_x)) if valid_x is not None else bg_noise_valid_x
            validation_labels += [class_list['_background_noise_']] * len(bg_noise_valid_x)

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
            logger.error('Only v2 variant supported')
            raise ValueError

        if self.__subset == 'digits':
            class_list = GSC.class_list_digits
        elif self.__subset == 'no_background_noise':
            class_list = GSC.class_list_no_background_noise
        elif self.__subset == 'tensorflow':
            class_list = GSC.class_list_tensorflow
        else:
            logger.error('Only digits, no_background_noise or tensorflow subsets supported')
            raise ValueError

        return self._load_v2(self._path, class_list=class_list)

    @property
    @override
    def name(self) -> str:
        return f'{super().name}_{self.__variant}_{self.__subset}'
