from __future__ import annotations

from abc import abstractmethod

from qualia_core.experimenttracking.ExperimentTracking import ExperimentTracking
from qualia_core.typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pytorch_lightning.loggers import Logger  # noqa: TCH002

class ExperimentTrackingPyTorch(ExperimentTracking):
    @property
    @abstractmethod
    def logger(self) -> Logger | None:
        ...
