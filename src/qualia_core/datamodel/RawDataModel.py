from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import astuple, dataclass
from typing import Any, Callable

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
    x: tuple[int, ...]
    y: tuple[int, ...]
    info: tuple[int, ...] | None = None

    @property
    def data(self) -> tuple[int, ...]:
        return self.x

    @data.setter
    def data(self, data: tuple[int, ...]) -> None:
        self.x = data

    @property
    def labels(self) -> tuple[int, ...]:
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
    shapes: RawDataShape  # We need to keep track of the global shape to pre-allocate mmapped-output file
    dtypes: RawDataDType  # We need to keep track of the global dtype to pre-allocate mmapped-output file

    def export(self, path: Path) -> None:
        start = time.time()

        data_file: np.ndarray[Any, Any] = np.lib.format.open_memmap(path / 'data.npy',
                                                                    mode='w+',
                                                                    dtype=self.dtypes.data,
                                                                    shape=self.shapes.data)
        labels_file: np.ndarray[Any, Any] = np.lib.format.open_memmap(path / 'labels.npy',
                                                                    mode='w+',
                                                                    dtype=self.dtypes.labels,
                                                                    shape=self.shapes.labels)
        if self.dtypes.info is not None and self.shapes.info is not None:
            info_file: np.ndarray[Any, Any] | None = np.lib.format.open_memmap(path / 'info.npy',
                                                                               mode='w+',
                                                                               dtype=self.dtypes.info,
                                                                               shape=self.shapes.info)
        else:
            info_file = None

        data_offset = 0
        labels_offset = 0
        info_offset = 0

        for chunk in self.chunks:
            data_file[data_offset:data_offset + chunk.data.shape[0]] = chunk.data
            data_offset += chunk.data.shape[0]

            labels_file[labels_offset:labels_offset + chunk.labels.shape[0]] = chunk.labels
            labels_offset += chunk.labels.shape[0]

            if chunk.info and info_file:
                info_file[info_offset:info_offset + chunk.info.shape[0]] = chunk.info
                info_offset += chunk.info.shape[0]

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
