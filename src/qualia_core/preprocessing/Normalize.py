from __future__ import annotations

import logging
import sys
from abc import abstractmethod
from collections.abc import Iterable
from typing import Any, ClassVar

import numpy as np

from qualia_core.datamodel.RawDataModel import RawData, RawDataModel

from .Preprocessing import Preprocessing, iterate_generator

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override


class NormalizeMethod(Preprocessing[RawDataModel, RawDataModel]):
    def __init__(self,
                 axis: int | list[int] | None = None,
                 debug: bool = False) -> None:  # noqa: FBT001, FBT002
        super().__init__()
        if axis is None:
            self._axis = (0,)
        elif isinstance(axis, Iterable):
            self._axis = tuple(axis)
        else:
            self._axis = (axis,)

        self.logger = logging.getLogger(f'{__name__}.{id(self)}')

        if debug:
            self.logger.setLevel(logging.DEBUG)

    def _print_dataset_stats(self, s: RawData, sname: str) -> None:
        self.logger.debug('%s: min=%s, max=%s, mean=%s, std=%s', sname, s.x.min(), s.x.max(), s.x.mean(), s.x.std())

    @abstractmethod
    @iterate_generator
    def _method(self, s: RawData, sname: str) -> RawData:
        """Normalize by chunk the train dataset but keep track of the global statistics to normalize test dataset.

        Requires the train dataset to be processed first, then valid and test."""
        raise NotImplementedError

    @override
    def __call__(self, datamodel: RawDataModel) -> RawDataModel:
        for sname, s in datamodel:
            setattr(datamodel.sets, sname, self._method(s, sname))

        return datamodel

class NormalizeZScore(NormalizeMethod):
    __train_x_mean: np.ndarray[Any, np.dtype[np.float64]]
    __train_x_squared_mean: np.ndarray[Any, np.dtype[np.float64]]
    __train_x_count: int = 0

    @override
    @iterate_generator
    def _method(self, s: RawData, sname: str) -> RawData:
        self.logger.debug('Before normalization')
        self._print_dataset_stats(s, sname)

        x_mean = s.x.mean(axis=self._axis, keepdims=True)
        x_std = s.x.std(axis=self._axis, keepdims=True)

        if sname == 'train':
            # Compute E[X²] on current chunk
            x_squared_mean = (s.x * s.x).mean(axis=self._axis, keepdims=True)

            # Update global E[X]
            self.__train_x_mean = (self.__train_x_mean * self.__train_x_count + x_mean * s.x.shape[0])
            self.__train_x_mean /= (self.__train_x_count + s.x.shape[0])
            # Update global E[X²]
            self.__train_x_squared_mean = self.__train_x_squared_mean * self.__train_x_count + x_squared_mean * s.x.shape[0]
            self.__train_x_squared_mean /= (self.__train_x_count + s.x.shape[0])

            self.__train_x_count += s.x.shape[0]

            # Normalize current chunk with its stats (not global)
            s.x -= x_mean
            s.x /= x_std
        else:
            # Compute Var[X] = E[X²] - E[X]²
            train_x_var = self.__train_x_squared_mean - (self.__train_x_mean * self.__train_x_mean)
            # Compute σ = √Var[X]
            train_x_std = np.sqrt(train_x_var)

            # Normalize current test/valid chunk with global stats
            s.x -= self.__train_x_mean
            s.x /= train_x_std

        self.logger.debug('After normalization')
        self._print_dataset_stats(s, sname)
        return s

    def __init__(self,
                 axis: int | list[int] | None = None,
                 debug: bool = False) -> None:  # noqa: FBT001, FBT002
        super().__init__(axis=axis, debug=debug)
        self.__train_x_mean = np.zeros((1), dtype=np.float64)
        self.__train_x_squared_mean = np.zeros((1), dtype=np.float64)
        self.__train_x_count = 0


class NormalizeMinMax(NormalizeMethod):
    __train_x_min: np.ndarray[Any, np.dtype[np.float32]]
    __train_x_max: np.ndarray[Any, np.dtype[np.float32]]

    @override
    @iterate_generator
    def _method(self, s: RawData, sname: str) -> RawData:
        self.logger.debug('Before normalization')
        self._print_dataset_stats(s, sname)

        x_min = s.x.min(axis=tuple(self._axis), keepdims=True)
        x_max = s.x.max(axis=tuple(self._axis), keepdims=True)

        if sname == 'train':
            # Update global min/max
            self.__train_x_min = np.minimum(self.__train_x_min, x_min)
            self.__train_x_max = np.maximum(self.__train_x_max, x_max)

            # Normalize current chunk with its stats (not global)
            s.x -= x_min
            s.x /= (x_max - x_min)
        else:
            # Normalize current test/valid chunk with global stats
            s.x -= self.__train_x_min
            s.x /= (self.__train_x_max - self.__train_x_min)

        self.logger.debug('After normalization')
        self._print_dataset_stats(s, sname)
        return s

    def __init__(self,
                 axis: int | list[int] | None = None,
                 debug: bool = False) -> None:  # noqa: FBT001, FBT002
        super().__init__(axis=axis, debug=debug)
        self.__train_x_min = np.full((1), np.inf, dtype=np.float32)
        self.__train_x_max = np.full((1), -np.inf, dtype=np.float32)


class Normalize(Preprocessing[RawDataModel, RawDataModel]):

    methods: ClassVar[dict[str, type[NormalizeMethod]]] = {
        'z-score': NormalizeZScore,
        'min-max': NormalizeMinMax,
    }

    def __init__(self,
                 method: str = 'z-score',
                 axis: int | list[int] | None = None,
                 debug: bool = False) -> None:  # noqa: FBT001, FBT002
        super().__init__()

        self.logger = logging.getLogger(f'{__name__}.{id(self)}')

        if debug:
            self.logger.setLevel(logging.DEBUG)

        if method not in self.methods:
            self.logger.error('Method %s is not supported. Supported methods: %s', method, ', '.join(self.methods))
            raise ValueError

        self.__method = self.methods[method](axis=axis, debug=debug)

    @override
    def __call__(self, datamodel: RawDataModel) -> RawDataModel:
        return self.__method(datamodel)
