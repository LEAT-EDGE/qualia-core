from __future__ import annotations

from abc import ABC, abstractmethod


class QuantizedLayer(ABC):
    @property
    @abstractmethod
    def input_q(self) -> int | None:
        ...

    @property
    @abstractmethod
    def activation_q(self) -> int | None:
        ...

    @property
    @abstractmethod
    def weights_q(self) -> int | None:
        ...
