import asyncio
import logging
import os
import sys
from typing import BinaryIO, Optional

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)

class AsyncTeeProtocol(asyncio.SubprocessProtocol):
    transport: Optional[asyncio.SubprocessTransport] = None

    def __init__(self,
                 done_future: asyncio.Future[tuple[int, dict[int, bytearray]]],
                 files: Optional[dict[int, BinaryIO]] = None) -> None:
        super().__init__()
        self.done = done_future
        self.buffers: dict[int, bytearray] = {}
        self.files = files if files is not None else {}

    @override
    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        if isinstance(transport, asyncio.SubprocessTransport):
            self.transport = transport
        else:
            logger.error('Expected transport type asyncio.SubprocessTransport, got: %s', type(asyncio.SubprocessTransport))


    @override
    def pipe_data_received(self, fd: int, data: bytes) -> None:
        _ = os.write(fd, data)
        if fd not in self.buffers:
            self.buffers[fd] = bytearray()
        self.buffers[fd].extend(data)
        if fd in self.files:
            _ = self.files[fd].write(data)

    @override
    def process_exited(self) -> None:
        if self.transport is None:
            logger.error('Transport is None')
        else:
            return_code = self.transport.get_returncode()
            if return_code is None:
                logger.error('return_code is None')
            else:
                self.done.set_result((return_code, self.buffers))
