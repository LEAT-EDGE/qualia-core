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
    from pathlib import Path  # noqa: TC003

    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self

logger = logging.getLogger(__name__)


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
    @classmethod
    def import_data(cls,
                    name: str,
                    set_names: list[str] | None = None,
                    sets_cls: type[DataModel.Sets[RawData]] = RawDataSets,
                    importer: Callable[[Path], RawData | None] = RawData.import_data) -> RawDataModel | None:
        set_names = set_names if set_names is not None else list(RawDataSets.fieldnames())

        return super().import_data(name=name, set_names=set_names, sets_cls=sets_cls, importer=importer)
