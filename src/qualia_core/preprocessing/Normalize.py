from __future__ import annotations

import logging
import sys
from typing import Callable, ClassVar

from qualia_core.datamodel.RawDataModel import RawDataModel

from .Preprocessing import Preprocessing

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override


class Normalize(Preprocessing[RawDataModel, RawDataModel]):
    def __print_dataset_stats(self, dataset: RawDataModel) -> None:
        for name, s in dataset.sets:
            self.logger.debug('%s: min=%s, max=%s, mean=%s, std=%s', name, s.x.min(), s.x.max(), s.x.mean(), s.x.std())

    def z_score(self, dataset: RawDataModel) -> RawDataModel:
        x_mean = dataset.sets.train.x.mean(axis=self.__axis, keepdims=True)
        x_std = dataset.sets.train.x.std(axis=self.__axis, keepdims=True)

        for _, s in dataset:
            s.x -= x_mean
            s.x /= x_std

        return dataset

    def min_max(self, datamodel: RawDataModel) -> RawDataModel:
        x_min = datamodel.sets.train.x.min(axis=tuple(self.__axis), keepdims=True)
        x_max = datamodel.sets.train.x.max(axis=tuple(self.__axis), keepdims=True)

        for _, s in datamodel:
            s.x -= x_min
            s.x /= (x_max - x_min)

        return datamodel

    methods: ClassVar[dict[str, Callable[[Self, RawDataModel], RawDataModel]]] = {
        'z-score': z_score,
        'min-max': min_max,
    }

    def __init__(self,
                 method: str = 'z-score',
                 axis: tuple[int, ...] = (0,),
                 debug: bool = False) -> None:  # noqa: FBT001, FBT002
        super().__init__()

        self.logger = logging.getLogger(f'{__name__}.{id(self)}')

        if debug:
            self.logger.setLevel(logging.DEBUG)

        if method not in self.methods:
            self.logger.error('Method %s is not supported. Supported methods: %s', method, ', '.join(self.methods))
            raise ValueError

        self.__method = self.methods[method].__get__(self)
        self.__axis = axis

    @override
    def __call__(self, datamodel: RawDataModel) -> RawDataModel:
        self.logger.debug('Before normalization')
        self.__print_dataset_stats(datamodel)

        datamodel = self.__method(datamodel)

        self.logger.debug('After normalization')
        self.__print_dataset_stats(datamodel)

        return datamodel
