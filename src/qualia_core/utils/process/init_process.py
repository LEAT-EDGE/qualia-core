from __future__ import annotations

import os


def init_process() -> None:
    # On Windows, we need to call setup_root_logger again since it creates a new process with CreateProcess()
    # instead of using fork() as on POSIX.
    if os.name == 'nt':
        from qualia_core.utils.logger.setup_root_logger import setup_root_logger

        setup_root_logger(colored=True)
