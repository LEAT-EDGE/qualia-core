"""Path utilities."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from qualia_core.typing import TYPE_CHECKING

# We are inside a TYPE_CHECKING block but our custom TYPE_CHECKING constant triggers TCH001-TCH003 so ignore them
if TYPE_CHECKING:
    from importlib.abc import Traversable  # noqa: TCH003
    from importlib.readers import MultiplexedPath  # noqa: TCH003


logger = logging.getLogger(__name__)

def resources_to_path(resources: Path | MultiplexedPath | Traversable) -> Path:
    """Convert a (static type) Traversable to a Path if the underlying dynamic type is MultiplexedPath.

    This is useful when using :func:`importlib.resources.files` since with Python >= 3.10 it returns a
    :class:`importlib.readers.MultiplexedPath` when :class:`importlib.readers.NamespaceReader` is used as a reader,
    e.g. when used on a package available as a plain directory (not zip file). This allows getting a standard :class:`pathlib.Path`
    out of it.
    Any other situation will probably fail with ValueError.

    :param resources: An object returned by :func:`importlib.resources.files`
    :return: A standard Path object pointing to the ``resources`` path on the filesystem
    :raise ValueError: When conversion fails because ``resources`` was not a :class:`importlib.readers.MultiplexedPath` on Python
        >= 3.10 or a :class:`pathlib.Path`
    """
    if isinstance(resources, Path): # Already Path objected, no need for hackery
        return resources

    if sys.version_info >= (3, 10): # Python 3.10 may return MultiplexedPath
        from importlib.readers import MultiplexedPath
        if isinstance(resources, MultiplexedPath):
            return resources / '' # / operator applies to underlying Path

    logger.error('Could not convert %s to Path object: expected type Path or MultiplexedPath for Python >= 3.10, got %s',
                 resources,
                 type(resources))
    raise ValueError
