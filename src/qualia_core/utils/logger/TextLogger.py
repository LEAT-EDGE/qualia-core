from __future__ import annotations

import logging
import sys
from typing import NamedTuple, overload

from qualia_core.typing import TYPE_CHECKING

from .Logger import Logger
from .TextFormatter import TextFormatter

if TYPE_CHECKING:
    from pathlib import Path  # noqa: TCH003

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

error_logger = logging.getLogger(__name__)

class TextLoggerField(NamedTuple):
    content: str

class TextLogger(Logger[TextLoggerField]):
    def __init__(self,
                 name: str,
                 file: Path | None = None) -> None:
        super().__init__(name=name, file=file, suffix='.txt', fields=TextLoggerField, log_fields=False, formatter=TextFormatter())

    @overload
    def __call__(self, data: str) -> None:
        ...

    @overload
    def __call__(self, data: TextLoggerField) -> None:
        ...

    @override
    def __call__(self, data: str | TextLoggerField) -> None:
        if self.logger is None:
            self._lazy_init()
        if self.logger is None:
            error_logger.error('Could not initialize logger %s', self._name)
            raise RuntimeError

        if not isinstance(data, TextLoggerField):
            self.logger.info(str(data))
            self._content.append(TextLoggerField(content=data))
        else:
            self.logger.info(data.content)
            self._content.append(data)
