from __future__ import annotations

from dataclasses import dataclass

from qualia_core.typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qualia_core.evaluation.Evaluator import Evaluator


@dataclass
class Deploy:
    rom_size: int | None
    ram_size: int | None
    evaluator: type[Evaluator]

