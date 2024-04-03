from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from qualia_core.dataset.Dataset import Dataset

T = TypeVar('T')
U = TypeVar('U')

class Preprocessing(ABC, Generic[T, U]):
    @abstractmethod
    def __call__(self, datamodel: T) -> U:
        ...

    def import_data(self, dataset: Dataset[Any]) -> Dataset[Any]:
        """no-op if the preprocessing doesn't modify the way of importing the dataset."""
        return dataset
