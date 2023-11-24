from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Generic

from qualia_core.learningframework.LearningFramework import LearningFramework, T
from qualia_core.typing import TYPE_CHECKING, ModelConfigDict

if TYPE_CHECKING:
    from qualia_core.qualia import TrainResult  # noqa: TCH001

class PostProcessing(ABC, Generic[T]):
    @abstractmethod
    def __call__(self, trainresult: TrainResult, model_conf: ModelConfigDict) -> tuple[TrainResult, ModelConfigDict]:
        ...

    def process_name(self, name: str) -> str:
        return name

    def process_framework(self, framework: LearningFramework[T]) -> LearningFramework[Any]:
        return framework

    def process_mem_params(self, mem_params: int) -> Callable[[LearningFramework[T],
                                                               T], int]:
        def f(_: LearningFramework[T], __: T) -> int:
            return mem_params
        return f

    def process_model(self, model: T, model_conf: ModelConfigDict) -> tuple[T, ModelConfigDict]:
        return model, model_conf
