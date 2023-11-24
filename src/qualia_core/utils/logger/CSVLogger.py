from __future__ import annotations

from qualia_core.typing import TYPE_CHECKING

from .CSVFormatter import CSVFormatter
from .Logger import Logger, T

if TYPE_CHECKING:
    from pathlib import Path  # noqa: TCH003


class CSVLogger(Logger[T]):
    def __init__(self,
                 name: str,
                 file: Path | None = None,
                 fields: type[T] | None = None,
                 log_fields: bool = True) -> None:  # noqa: FBT001, FBT002
        super().__init__(name=name, file=file, suffix='.csv', fields=fields, log_fields = log_fields, formatter=CSVFormatter())
