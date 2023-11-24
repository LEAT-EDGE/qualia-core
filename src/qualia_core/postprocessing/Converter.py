from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from qualia_core.typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType  # noqa: TCH003, I001 # torch must be imported before keras to avoid deadlock

    from torch import nn  # noqa: TCH002
    import keras  # type: ignore[import] # No stubs for keras package  # noqa: TCH002
    import numpy.typing  # noqa: TCH002

    from qualia_core.learningframework.LearningFramework import LearningFramework  # noqa: TCH001

class Converter(ABC):
    deployers: ModuleType | None

    @abstractmethod
    def convert(self,
                framework: LearningFramework[nn.Module | keras.Model],
                model: nn.Module | keras.Model,
                model_name: str,
                representative_dataset: numpy.typing.NDArray[Any]) -> Converter:
        ...
