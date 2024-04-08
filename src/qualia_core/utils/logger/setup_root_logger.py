from __future__ import annotations

import logging
import sys

from qualia_core.utils.logger.ConsoleFormatter import ConsoleFormatter


def setup_root_logger(colored: bool = False,  # noqa: FBT001, FBT002
                      level: int = logging.INFO) -> None:
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = ConsoleFormatter(fmt) if colored else logging.Formatter(fmt)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    out = logging.StreamHandler(sys.stdout)
    out.setFormatter(formatter)
    out.setLevel(logging.DEBUG)
    out.addFilter(lambda r: r.levelno <= logging.INFO)

    err = logging.StreamHandler(sys.stderr)
    err.setLevel(logging.WARNING)
    err.setFormatter(formatter)

    root_logger.addHandler(out)
    root_logger.addHandler(err)
