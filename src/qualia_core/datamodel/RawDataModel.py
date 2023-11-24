from __future__ import annotations

import logging
import os
import time
from dataclasses import astuple, dataclass
from pathlib import Path
from typing import Any

import blosc2
import numpy as np
import numpy.typing

from .DataModel import DataModel

logger = logging.getLogger(__name__)

@dataclass
class RawData:
    x: numpy.typing.NDArray[Any]
    y: numpy.typing.NDArray[Any]
    info: numpy.typing.NDArray[Any] | None = None

    @property
    def data(self) -> numpy.typing.NDArray[Any]:
        return self.x

    @data.setter
    def data(self, data: numpy.typing.NDArray[Any]) -> None:
        self.x = data

    @property
    def labels(self) -> numpy.typing.NDArray[Any]:
        return self.y

    @labels.setter
    def labels(self, labels: numpy.typing.NDArray[Any]) -> None:
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
    def import_data(cls, path: Path, compressed: bool = True) -> RawData | None:
        start = time.time()

        for fname in ['data.npz', 'labels.npz']:
            if not (path/fname).is_file():
                logger.error("'%s' not found. Did you run 'preprocess_data'?", path/fname)
                return None

        info: numpy.typing.NDArray[Any] | None = None

        if compressed:
            data: numpy.typing.NDArray[Any] = blosc2.load_array(str(path/'data.npz'))
            labels: numpy.typing.NDArray[Any] = blosc2.load_array(str(path/'labels.npz'))
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

    @classmethod
    def import_data(cls, name: str, sets: list[str] | None = None) -> RawDataModel | None:
        set_names = sets if sets is not None else list(RawDataSets.fieldnames())
        sets_dict: dict[str, RawData | None] =  {sname: RawData.import_data(Path('out')/'data'/name/sname)
                                                   for sname in set_names}

        if any(s is None for s in sets_dict.values()):
            logger.error('Could not import data.')
            return None

        logger.info('Imported %s for %s', ', '.join(sets_dict.keys()), name)

        return cls(sets=RawDataSets(**sets_dict),
                   name=name)
