from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from qualia_core.typing import TYPE_CHECKING, OptimizerConfigDict

if TYPE_CHECKING:
    from types import ModuleType  # noqa: TCH003

    import numpy.typing  # noqa: TCH002

    from qualia_core.dataaugmentation.DataAugmentation import DataAugmentation  # noqa: TCH001
    from qualia_core.datamodel.RawDataModel import RawData  # noqa: TCH001
    from qualia_core.experimenttracking.ExperimentTracking import ExperimentTracking  # noqa: TCH001

T = TypeVar('T')

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
    def load(self, name: str, model: T) -> T:
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
    def export(self, model: T, name: str) -> None:
        pass

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
