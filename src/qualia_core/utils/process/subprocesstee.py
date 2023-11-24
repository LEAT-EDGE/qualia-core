import asyncio
import os
from pathlib import Path
from typing import BinaryIO, Optional, TextIO, Union

from .AsyncTeeProtocol import AsyncTeeProtocol


async def async_exec(*argv: str,
                     files: Optional[dict[int, BinaryIO]] = None,
                     cwd: Optional[Path] = None,
                     env: Optional[dict[str, str]] = None) -> tuple[int, dict[int, bytearray]]:
    files = files if files is not None else {}
    loop = asyncio.get_running_loop()
    cmd_done = loop.create_future()
    proc = loop.subprocess_exec(lambda: AsyncTeeProtocol(cmd_done, files), *argv, stdin=None, cwd=cwd, env=env)
    transport = None
    try:
        transport, _ = await proc
        await cmd_done
    finally:
        if transport is not None:
            transport.close()

    return cmd_done.result()

def run(*argv: str,
        files: Optional[dict[Union[TextIO, BinaryIO], BinaryIO]] = None,
        cwd: Optional[Path] = None,
        env: Optional[dict[str, str]] = None) -> tuple[int, dict[int, bytearray]]:
    files = files if files is not None else {}
    env = env if env is not None else {}
    int_files = {fd.fileno(): f for fd, f in files.items()} # convert file objects to file descriptor
    return asyncio.run(async_exec(*argv, files=int_files, cwd=cwd, env={**os.environ, **env}))

