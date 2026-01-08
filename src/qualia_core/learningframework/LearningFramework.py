from __future__ import annotations

import logging
import hashlib
import sys
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from qualia_core.typing import TYPE_CHECKING, OptimizerConfigDict

if TYPE_CHECKING:
    from pathlib import Path
    from types import ModuleType

    import numpy.typing

    from qualia_core.dataaugmentation.DataAugmentation import DataAugmentation
    from qualia_core.datamodel.RawDataModel import RawData
    from qualia_core.experimenttracking.ExperimentTracking import ExperimentTracking

T = TypeVar('T')

logger = logging.getLogger(__name__)


class LearningFramework(ABC, Generic[T]):
    learningmodels: ModuleType

    @abstractmethod
    def train(self,
              model: T,
              trainset: RawData,
              validationset: RawData,
              epochs: int,
              batch_size: int,
              optimizer: OptimizerConfigDict | None,
              dataaugmentations: list[DataAugmentation],
              experimenttracking: ExperimentTracking | None,
              name: str) -> T:
        pass

    @abstractmethod
    def load(self, name: str, model: T) -> tuple[T, Path]:
        pass

    @abstractmethod
    def evaluate(self,
                 model: T,
                 testset: RawData,
                 batch_size: int,
                 dataaugmentations: list[DataAugmentation],
                 experimenttracking: ExperimentTracking | None,
                 dataset_type: str,
                 name: str) -> dict[str, int | float | numpy.typing.NDArray[Any]]:
        pass

    @abstractmethod
    def predict(self,
                 model: T,
                 dataset: RawData,
                 batch_size: int,
                 dataaugmentations: list[DataAugmentation],
                 experimenttracking: ExperimentTracking | None,
                 name: str) -> Any:
        ...

    @abstractmethod
    def export(self, model: T, name: str) -> Path:
        pass

    def hash_model(self, path: Path) -> str | None:
        if not path.is_file():
            logger.error('%s not found, cannot compute hash.', path)
            return None

        # hashlib.file_digest() requires Python 3.11
        if sys.version_info < (3, 12):
            logger.error('Python 3.11 or newer required')
            raise NotImplementedError

        with path.open('rb') as f:
            filehash = hashlib.file_digest(f, 'sha256')
        return filehash.hexdigest()

    @abstractmethod
    def summary(self, model: T) -> None:
        pass

    @abstractmethod
    def n_params(self, model: T) -> int:
        pass

    @abstractmethod
    def save_graph_plot(self, model: T, model_save: str) -> None:
        pass

    @abstractmethod
    def apply_dataaugmentation(self,
                               da: DataAugmentation,
                               x: numpy.typing.NDArray[Any],
                               y: numpy.typing.NDArray[Any],
                               **kwargs: Any) -> tuple[numpy.typing.NDArray[Any], numpy.typing.NDArray[Any]]:
        ...
