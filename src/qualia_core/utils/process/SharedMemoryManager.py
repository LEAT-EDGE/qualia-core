from __future__ import annotations

import logging
import sys
from multiprocessing import shared_memory
from multiprocessing.managers import BaseProxy, Token, dispatch
from multiprocessing.managers import SharedMemoryManager as SharedMemoryManagerBase
from typing import Callable, cast

from qualia_core.typing import TYPE_CHECKING

if TYPE_CHECKING:
    from multiprocessing.connection import Connection  # noqa: TC003

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)

class SharedMemoryProxy(BaseProxy):
    _exposed_ = ('__getattribute__',)

    @property
    def name(self) -> str:
        return cast(str, self._callmethod('__getattribute__', ('name',)))

class SharedMemoryManager(SharedMemoryManagerBase):
    """Provides a SharedMemory factory that creates the object on the remote manager process instead of the current process.

    Used to workaround Windows-related issue where a SharedMemory object created in a child process would get destroyed and release
    the shared memory segment at the same time when the process ends, even though the parent process may still want to access it
    afterwards.
    """

    # Declare types of inherited attribute/methods that are not available in typeshed
    _address: str | tuple[str, int]
    _authkey: bytes
    _serializer: str
    _Client: Callable[..., Connection]
    _create: Callable[..., tuple[Token, bool]]

    @override
    def SharedMemory(self, size: int) -> shared_memory.SharedMemory:
        """Return a new SharedMemory instance with the specified size in bytes, created and tracked by the manager.

        :param size: Size of shared memory segment in bytes
        """
        address = self._address
        authkey = self._authkey
        logger.debug('Requesting creation of a SharedMemory object')
        # Create type on remote manager process
        token, exp = self._create('SharedMemory', create=True, size=size)
        proxy = SharedMemoryProxy(token,
                                  self._serializer,
                                  manager=self,
                                  authkey=authkey,
                                  exposed=exp,
                                  )
        with self._Client(address, authkey=authkey) as conn:
            dispatch(conn, None, 'decref', (token.id,))
        with self._Client(address, authkey=authkey) as conn:
            # Track SharedMemory object on remote manager process to manage its lifetime
            dispatch(conn, None, 'track_segment', (proxy.name,))
        # Return an actual SharedMemory object that uses the same segment name as the remote object
        return shared_memory.SharedMemory(proxy.name)

SharedMemoryManager.register('SharedMemory', shared_memory.SharedMemory, SharedMemoryProxy, create_method=False)
