from __future__ import annotations

import logging
import os
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

class SharedMemoryPersistent(shared_memory.SharedMemory):
    @override
    def close(self) -> None:
        """Make :meth:`shared_memory.SharedMemory.close` ineffective on Windows."""
        if os.name == 'nt':
            return

        super().close()

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

        Reference count on remote manager is not decremented to keep reference count above 1
        so that SharedMemory object does not get destroyed when child process exits
        and parent process has not accessed it yet. Otherwise, on Windows, the segment becomes inaccessible.


        The returned :class:`SharedMemoryPersistent` is a local :class:`shared_memory.SharedMemory` object
        that connects to the same segment as the remote object.
        On Windows, this local object has its :meth:`SharedMemoryPersistent.close` method ineffective on Windows
        in order to prevent deletion of the shared segment when the child process exits.

        The SharedMemory object is still tracked with track_segment so that :meth:`close()` and unlink() get called
        when the SharedMemoryManager gets destroyed.

        :param size: Size of shared memory segment in bytes
        """
        logger.debug('Requesting creation of a SharedMemory object')

        # Create object on remote manager process
        token, exp = self._create('SharedMemory', create=True, size=size)
        proxy = SharedMemoryProxy(token,
                                  self._serializer,
                                  manager=self,
                                  authkey=self._authkey,
                                  exposed=exp,
                                  )

        # Track SharedMemory object on remote manager process to manage its lifetime
        with self._Client(self._address, authkey=self._authkey) as conn:
            dispatch(conn, None, 'track_segment', (proxy.name,))

        return SharedMemoryPersistent(proxy.name)

SharedMemoryManager.register('SharedMemory', shared_memory.SharedMemory, SharedMemoryProxy, create_method=False)
