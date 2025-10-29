from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import astuple, dataclass
from typing import Any, Callable, Final

import blosc2
import numpy as np

from qualia_core.typing import TYPE_CHECKING

from .DataModel import DataModel

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self

logger = logging.getLogger(__name__)


@dataclass
class RawDataShape:
    x: tuple[int | None, ...]
    y: tuple[int | None, ...]
    info: tuple[int | None, ...] | None = None

    @property
    def data(self) -> tuple[int | None, ...]:
        return self.x

    @data.setter
    def data(self, data: tuple[int, ...]) -> None:
        self.x = data

    @property
    def labels(self) -> tuple[int | None, ...]:
        return self.y

    @labels.setter
    def labels(self, labels: tuple[int, ...]) -> None:
        self.y = labels


@dataclass
class RawDataDType:
    x: np.dtype
    y: np.dtype
    info: np.dtype | None = None

    @property
    def data(self) -> np.dtype:
        return self.x

    @data.setter
    def data(self, data: np.dtype) -> None:
        self.x = data

    @property
    def labels(self) -> np.dtype:
        return self.y

    @labels.setter
    def labels(self, labels: np.dtype) -> None:
        self.y = labels


@dataclass
class RawDataChunks:
    chunks: Generator[RawData]
    # Pre-define the global shape to check consistency and proper allocation of output mmapped-file.
    # First dimension (total numberof samples) is left undefined and set to ``None``
    shapes: RawDataShape
    # Pre-define the global dtype to check consistency and proper allocation of output mmapped-file
    dtypes: RawDataDType

    @staticmethod
    def __check_shape(shape1: tuple[int | None, ...], shape2: tuple[int | None, ...]) -> bool:
        """Check shape for defined dimensions.

        Dimensions set to ``None`` are undefined and not checked
        :return: ``True`` if defined dimensions match, ``False`` otherwise
        """
        return all(s1 == s2 for s1, s2 in zip(shape1, shape2) if s1 is not None and s2 is not None)

    def __write_array(self,
                      path: Path,
                      array: np.ndarray[Any, Any],
                      shape: tuple[int, ...],
                      dtype: np.dtype,
                      offset: int) -> int:
        if not self.__check_shape(array.shape, shape):
            logger.error('Chunk array shape %s does not match dataset shape %s for file %s', array.shape, shape, path)
            raise ValueError

        if array.dtype != dtype:
            logger.error('Chunk array dtype %s does not match dataset dtype %s for file %s', array.dtype, dtype, path)
            raise ValueError

        data_file = np.memmap(path,
                                dtype=array.dtype,
                                mode='r+',  # Write without truncate
                                offset=offset,
                                shape=array.shape)
        data_file[:] = array
        offset += array.nbytes
        data_file._mmap.close()

        return offset

    @staticmethod
    def __write_numpy_header(path: Path, header_size: int, shape: tuple[int, ...], dtype: np.dtype) -> None:
        with (path).open('r+b') as f:
            np.lib.format.write_array_header_1_0(f, {'shape': shape,
                                                     'fortran_order': False,
                                                     'descr': np.dtype(dtype).str})
            if f.tell() != header_size:
                logger.error('NumPy header of size %d different from pre-allocated size of %d for file %s',
                             f.tell(),
                             header_size,
                             path)
                raise RuntimeError

    def export(self, path: Path) -> None:
        NUMPY_HEADER_SIZE: Final[int] = 128

        start = time.time()

        files = ['data', 'labels']
        if self.shapes.info:
            files.append('info')

        # Position in the output file
        offsets = dict.fromkeys(files, NUMPY_HEADER_SIZE)
        # Total number of samples
        counts = dict.fromkeys(files, 0)

        # Create empty files or truncate existing files
        for file in files:
            with (path / f'{file}.npy').open('wb'):
                pass

        # Iterate over dataset generator, actually calling preprocessing pipeline for each chunk, and write results to files
        for chunk in self.chunks:
            for file in files:
                offsets[file] = self.__write_array(path / f'{file}.npy',
                                                    getattr(chunk, file),
                                                    shape=getattr(self.shapes, file),
                                                    dtype=getattr(self.dtypes, file),
                                                    offset=offsets[file])
                counts[file] += len(chunk.data)

        # Write NumPy headers after obtaining total number of samples
        for file in files:
            shape = (counts[file], *getattr(self.shapes, file)[1:])
            self.__write_numpy_header(path / f'{file}.npy',
                                      header_size=NUMPY_HEADER_SIZE,
                                      shape=shape,
                                      dtype=getattr(self.dtypes, file))

        logger.info('export() Elapsed: %s s', time.time() - start)

    @classmethod
    def import_data(cls, path: Path) -> RawData | None:
        start = time.time()

        for fname in ['data.npy', 'labels.npy']:
            if not (path / fname).is_file():
                logger.error("'%s' not found. Did you run 'preprocess_data'?", path / fname)
                return None

        info: np.ndarray[Any, Any] | None = None

        data = np.load(path / 'data.npy', mmap_mode='r')
        labels = np.load(path / 'labels.npy', mmap_mode='r')
        if (path / 'info.npy').is_file():
            info = np.load(path / 'info.npy', mmap_mode='r')

        ret = RawData(x=data, y=labels, info=info)
        logger.info('import_data() Elapsed: %s s', time.time() - start)
        return ret

    def astuple(self) -> tuple[Any, ...]:
        return astuple(self)


@dataclass
class RawData:
    x: np.ndarray[Any, Any]
    y: np.ndarray[Any, Any]
    info: np.ndarray[Any, Any] | None = None

    @property
    def data(self) -> np.ndarray[Any, Any]:
        return self.x

    @data.setter
    def data(self, data: np.ndarray[Any, Any]) -> None:
        self.x = data

    @property
    def labels(self) -> np.ndarray[Any, Any]:
        return self.y

    @labels.setter
    def labels(self, labels: np.ndarray[Any, Any]) -> None:
        self.y = labels

    def export(self, path: Path, compressed: bool = True) -> None:
        start = time.time()
        if compressed:
            cparams = {'codec': blosc2.Codec.ZSTD, 'clevel': 5, 'nthreads': os.cpu_count()}
            blosc2.pack_array2(np.ascontiguousarray(self.data), urlpath=str(path/'data.npz'), mode='w', cparams=cparams)
            blosc2.pack_array2(np.ascontiguousarray(self.labels), urlpath=str(path/'labels.npz'), mode='w', cparams=cparams)
            if self.info is not None:
                blosc2.pack_array2(np.ascontiguousarray(self.info), urlpath=str(path/'info.npz'), mode='w', cparams=cparams)
        else:
            np.savez(path/'data.npz', data=self.data)
            np.savez(path/'labels.npz', labels=self.labels)
            if self.info is not None:
                np.savez(path/'info.npz', info=self.info)
        logger.info('export() Elapsed: %s s', time.time() - start)

    @classmethod
    def import_data(cls, path: Path, compressed: bool = True) -> Self | None:
        start = time.time()

        for fname in ['data.npz', 'labels.npz']:
            if not (path/fname).is_file():
                logger.error("'%s' not found. Did you run 'preprocess_data'?", path/fname)
                return None

        info: np.ndarray[Any, Any] | None = None

        if compressed:
            data: np.ndarray[Any, Any] = blosc2.load_array(str(path/'data.npz'))
            labels: np.ndarray[Any, Any] = blosc2.load_array(str(path/'labels.npz'))
            if (path/'info.npz').is_file():
                info = blosc2.load_array(str(path/'info.npz'))
        else:
            with np.load(path/'data.npz') as datanpz:
                data = datanpz['data']
            with np.load(path/'labels.npz') as labelsnpz:
                labels = labelsnpz['labels']

            if (path/'info.npz').is_file():
                with np.load(path/'info.npz') as infonpz:
                    info = infonpz['info']

        ret = cls(x=data, y=labels, info=info)
        logger.info('import_data() Elapsed: %s s', time.time() - start)
        return ret

    def astuple(self) -> tuple[Any, ...]:
        return astuple(self)


class RawDataSets(DataModel.Sets[RawData]):
    ...


class RawDataModel(DataModel[RawData]):
    sets: DataModel.Sets[RawData]

    @override
    def import_sets(self,
                    set_names: list[str] | None = None,
                    sets_cls: type[DataModel.Sets[RawData]] = RawDataSets,
                    importer: Callable[[Path], RawData | None] = RawData.import_data) -> None:
        set_names = set_names if set_names is not None else list(RawDataSets.fieldnames())

        sets_dict = self._import_data_sets(name=self.name, set_names=set_names, importer=importer)

        if sets_dict is not None:
            self.sets = sets_cls(**sets_dict)


class RawDataChunksSets(DataModel.Sets[RawDataChunks]):
    ...


class RawDataChunksModel(DataModel[RawDataChunks]):
    sets: DataModel.Sets[RawDataChunks]

    @override
    def import_sets(self,
                    set_names: list[str] | None = None,
                    sets_cls: type[DataModel.Sets[RawDataChunks]] = RawDataChunksSets,
                    importer: Callable[[Path], RawData | None] = RawDataChunks.import_data) -> None:
        set_names = set_names if set_names is not None else list(RawDataSets.fieldnames())

        sets_dict = self._import_data_sets(name=self.name, set_names=set_names, importer=importer)

        if sets_dict is not None:
            self.sets = sets_cls(**sets_dict)
