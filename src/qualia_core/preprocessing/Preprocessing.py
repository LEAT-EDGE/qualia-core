from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar, Union

from qualia_core.datamodel.RawDataModel import RawData, RawDataChunks, RawDataModel

if TYPE_CHECKING:
    from qualia_core.dataset.Dataset import Dataset

if sys.version_info >= (3, 10):
    from typing import Concatenate, ParamSpec
else:
    from typing_extensions import ParamSpec, Concatenate

P = ParamSpec('P')
T = TypeVar('T')
U = TypeVar('U')

IterateGeneratorCallable = Callable[Concatenate[Any, RawData, P], RawData]
IterateGeneratorCallableDecorated = Callable[Concatenate[Any, Union[RawData, RawDataChunks], P], Union[RawData, RawDataChunks]]


def iterate_generator(f: IterateGeneratorCallable[P]) -> IterateGeneratorCallableDecorated[P]:
    def decorated(self: Preprocessing[RawDataModel, RawDataModel],
                  s: RawDataChunks | RawData,
                  *args: P.args,
                  **kwargs: P.kwargs) -> RawData | RawDataChunks:
        if isinstance(s, RawDataChunks):
            s.chunks = (f(self, chunk, *args, **kwargs) for chunk in s.chunks)
            return s

        # Not a Generator
        return f(self, s, *args, **kwargs)
    return decorated


class Preprocessing(ABC, Generic[T, U]):
    @abstractmethod
    def __call__(self, datamodel: T) -> U:
        ...

    def import_data(self, dataset: Dataset[Any]) -> Dataset[Any]:
        """no-op if the preprocessing doesn't modify the way of importing the dataset."""
        return dataset
