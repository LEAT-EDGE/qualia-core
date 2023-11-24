from __future__ import annotations

from dataclasses import dataclass

from qualia_core.typing import TYPE_CHECKING

# We are inside a TYPE_CHECKING block but our custom TYPE_CHECKING constant triggers TCH001-TCH003 so ignore them
if TYPE_CHECKING:
    from qualia_core.evaluation.Evaluator import Evaluator  # noqa: TCH001


@dataclass
class Deploy:
    rom_size: int | None
    ram_size: int | None
    evaluator: type[Evaluator]

