from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from qualia_core.datamodel.DataModel import DataModel

T = TypeVar('T')

class Dataset(ABC, Generic[T]):
    sets: list[str]

    def __init__(self, sets: list[str] | None = None) -> None:
        super().__init__()
        self.sets = sets if sets is not None else list(DataModel.Sets.fieldnames())

    @abstractmethod
    def __call__(self) -> DataModel[T]:
        ...

    @abstractmethod
    def import_data(self) -> DataModel[T] | None:
        ...

    @property
    def name(self) -> str:
        return f'{self.__class__.__name__}'
