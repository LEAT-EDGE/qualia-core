from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from qualia_core.typing import TYPE_CHECKING

# We are inside a TYPE_CHECKING block but our custom TYPE_CHECKING constant triggers TCH001-TCH003 so ignore them
if TYPE_CHECKING:
    from qualia_core.dataaugmentation.DataAugmentation import DataAugmentation  # noqa: TCH001
    from qualia_core.datamodel.RawDataModel import RawDataModel  # noqa: TCH001
    from qualia_core.evaluation.Stats import Stats  # noqa: TCH001
    from qualia_core.learningframework.LearningFramework import LearningFramework  # noqa: TCH001


class Evaluator(ABC):
    @abstractmethod
    def evaluate(self,  # noqa: PLR0913
                 framework: LearningFramework[Any],
                 model_kind: str,
                 dataset: RawDataModel,
                 target: str,
                 tag: str,
                 limit: int | None = None,
                 dataaugmentations: list[DataAugmentation] | None = None) -> Stats | None:
        ...
