from abc import ABC, abstractmethod
from typing import Optional

from qualia_core.typing import RecursiveConfigDict


class ExperimentTracking(ABC):
    @abstractmethod
    def start(self, name: Optional[str] = None) -> None:
        ...

    @abstractmethod
    def stop(self) -> None:
        ...

    @classmethod
    def initializer(cls) -> None:
        pass

    @abstractmethod
    def _hyperparameters(self, params: RecursiveConfigDict) -> None:
        ...

    # Lambda used to obtain concrete property from abstract setter
    hyperparameters = property(None, lambda self, x: self._hyperparameters(x))
