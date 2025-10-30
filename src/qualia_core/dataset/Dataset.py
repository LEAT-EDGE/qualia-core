from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from typing import Generic

from qualia_core.datamodel.DataModel import DataModel

if sys.version_info >= (3, 13):
    from typing import TypeVar
else:
    from typing_extensions import TypeVar

T = TypeVar('T')
# Dataset.import_data() may return a different DataModel than Dataset.__call__(), e.g., non-chunked
U = TypeVar('U', default=T)


class Dataset(ABC, Generic[T, U]):
    sets: list[str]

    def __init__(self, sets: list[str] | None = None) -> None:
        super().__init__()
        self.sets = sets if sets is not None else list(DataModel.Sets.fieldnames())

    @abstractmethod
    def __call__(self) -> DataModel[T, U]:
        ...

    @abstractmethod
    def import_data(self) -> DataModel[U] | None:
        ...

    @property
    def name(self) -> str:
        return f'{self.__class__.__name__}'
