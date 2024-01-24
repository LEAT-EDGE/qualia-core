from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import Final

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

@dataclass
class ConsoleColors:
    #GREY = '\x1b[30;1m'
    CYAN = '\x1b[36m'
    BLUE = '\x1b[34m'
    YELLOW = '\x1b[33m'
    RED = '\x1b[31m'
    BROWN = '\x1b[31;1m'
    RESET = '\x1b[0m'


class ConsoleFormatter(logging.Formatter):

    FORMATS: Final[dict[int, str]] = {
        logging.DEBUG: ConsoleColors.CYAN,
        logging.INFO: ConsoleColors.RESET,
        logging.WARNING: ConsoleColors.YELLOW,
        logging.ERROR: ConsoleColors.RED,
        logging.CRITICAL: ConsoleColors.BROWN,
    }

    @override
    def format(self, record: logging.LogRecord) -> str:
        return self.FORMATS[record.levelno] + super().format(record) + ConsoleColors.RESET
