from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Generic

if TYPE_CHECKING:
    from collections.abc import Iterator

    if sys.version_info >= (3, 10):
        from typing import TypeGuard
    else:
        from typing_extensions import TypeGuard

if sys.version_info >= (3, 13):
    from typing import TypeVar
else:
    from typing_extensions import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')
U = TypeVar('U')
# DataModel.import_sets() may return a different DataModel than itself, e.g., non-chunked
V = TypeVar('V', default=T)


class DataModel(Generic[T, V]):
    sets: DataModel.Sets[T]
    name: str

    @dataclass
    class Sets(Generic[U]):
        train: U | None = None
        valid: U | None = None
        test: U | None = None

        @classmethod
        def fieldnames(cls) -> tuple[str, ...]:
            return tuple(f.name for f in fields(cls))

        def asdict(self) -> dict[str, U]:
            # Explicitely use vars rather than dataclasses.asdict() since we don't want to recurse nor copy
            return vars(self)

        def __iter__(self) -> Iterator[tuple[str, U]]:
            # Skip non-existent sets
            return {k: v for k, v in self.asdict().items() if v is not None}.items().__iter__()

        def export(self, path: Path) -> None:
            for sname, sdata in self.asdict().items():
                if not sdata:
                    logger.info('No "%s" set to export', sname)
                else:
                    (path/sname).mkdir(parents=True, exist_ok=True)
                    sdata.export(path/sname)

    def __init__(self, name: str, sets: Sets[T] | None = None) -> None:
        super().__init__()
        self.sets = sets if sets is not None else self.Sets()
        self.name = name

    def __iter__(self) -> Iterator[tuple[str, T]]:
        return self.sets.__iter__()

    def export(self) -> None:
        self.sets.export(Path('out')/'data'/self.name)

    @classmethod
    def _import_data_sets(cls,
                    name: str,
                    set_names: list[str],
                    importer: Callable[[Path], V | None]) -> dict[str, V] | None:
        sets_dict: dict[str, V | None] = {sname: importer(Path('out')/'data'/name/sname)
                                                   for sname in set_names}

        def no_none_in_sets(sets_dict: dict[str, V | None]) -> TypeGuard[dict[str, V]]:
            return all(s is not None for s in sets_dict.values())

        if not no_none_in_sets(sets_dict):
            logger.error('Could not import data.')
            return None

        logger.info('Imported %s for %s', ', '.join(sets_dict.keys()), name)
        return sets_dict

    def import_sets(self,
                    set_names: list[str],
                    sets_cls: type[DataModel.Sets[V]],
                    importer: Callable[[Path], V | None]) -> DataModel[V, Any]:
        sets_dict = self._import_data_sets(name=self.name, set_names=set_names, importer=importer)

        if sets_dict is not None:
            self.sets = sets_cls(**sets_dict)

        return self
