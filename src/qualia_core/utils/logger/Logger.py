from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Generic, NamedTuple, TypeVar

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

error_logger = logging.getLogger(__name__)

T = TypeVar('T', bound=NamedTuple)

class Logger(Generic[T]):
    logpath = Path('logs')
    prefix = ''

    def __init__(self,
                 name: str,
                 file: Path | None = None,
                 suffix: str = '.log',
                 fields: type[T] | None = None,
                 log_fields: bool = True,
                 formatter: logging.Formatter | None = None) -> None:
        super().__init__()
        self._name = name.replace('.', '_') # module names contain dots but namedtuple requires identifier
        self.__file = file
        self.__suffix = suffix
        self.__formatter = formatter
        self.__init_fields = fields
        self.__fields: type[T] | None = None
        self.__log_fields = log_fields
        self._content: list[T] | list[tuple[str, ...]]= []
        self.logger: logging.Logger | None = None
        self.filehandler: logging.FileHandler | None = None

    def _lazy_init(self) -> None:
        """Lazy init allows setting configuring global parameters such as path/prefix even after instanciation
        (when classes with CSVLogger as class member are loaded) but before logger is called for the first time.
        """
        # Add logger to name to prevent conflict with other module-local loggers
        self.logger = logging.getLogger(f'qualia_core.logger.{self._name}')

        self.logger.propagate = False # Don't use logger hierarchy since somehow root logger may have a handler already
        self.logger.setLevel(logging.DEBUG)

        if not self.logger.hasHandlers():
            if self.__file is None:
                self.__file = Path(self._name)/f'{self.prefix}{datetime.now():%Y-%m-%d_%H-%M-%S}{self.__suffix}'
            self.__file = self.logpath/self.__file
            self.__file.parent.mkdir(parents=True, exist_ok=True)

            self.filehandler = logging.FileHandler(self.__file, delay=True)
            self.filehandler.setLevel(logging.DEBUG)
            if self.__formatter:
                self.filehandler.setFormatter(self.__formatter)
            self.logger.addHandler(self.filehandler)
        if self.__init_fields is not None:
            self.fields = self.__init_fields

    def __call__(self, data: T) -> None:
        if self.logger is None:
            self._lazy_init()
        if self.logger is None:
            error_logger.error('Could not initialize logger %s', self._name)
            raise RuntimeError

        str_args = tuple(str(v) for v in data)

        self.logger.info(str_args)

        if self.fields is not None:
            self._content.append(data)
        else:
            self._content.append(str_args)

    def __del__(self) -> None:
        if self.filehandler is not None:
            self.filehandler.close()
            if self.logger is not None:
                self.logger.removeHandler(self.filehandler)

    @override
    def __repr__(self) -> str:
        return str(self.content)

    @property
    def fields(self) -> type[T] | None:
        return self.__fields

    @fields.setter
    def fields(self, val: type[T]) -> None:
        if self.__fields is not None:
            error_logger.error("'fields' can only be assigned once for each instance of %s", type(self))
            raise AttributeError
        self.__fields = val

        # Write column names to file
        if self.logger is None:
            self._lazy_init()
        if self.logger is None:
            error_logger.error('Could not initialize logger %s', self._name)
            raise RuntimeError

        if self.__log_fields:
            self.logger.info(val._fields)

    @property
    def content(self) -> list[T] | list[tuple[str, ...]]:
        return self._content
