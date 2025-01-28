from __future__ import annotations

import logging
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Generic, TypeVar

if TYPE_CHECKING:
    import sys
    from collections.abc import Iterator

    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self

logger = logging.getLogger(__name__)

T = TypeVar('T')
U = TypeVar('U')

class DataModel(Generic[T]):
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
            return {k: v for k,v in self.asdict().items() if v is not None}.items().__iter__()

        def export(self, path: Path) -> None:
            for sname, sdata in self.asdict().items():
                if not sdata:
                    logger.info('No "%s" set to export', sname)
                else:
                    (path/sname).mkdir(parents=True, exist_ok=True)
                    sdata.export(path/sname)

    def __init__(self, sets: Sets[T], name: str) -> None:
        super().__init__()
        self.sets = sets
        self.name = name

    def __iter__(self) -> Iterator[tuple[str, T]]:
        return self.sets.__iter__()

    def export(self) -> None:
        self.sets.export(Path('out')/'data'/self.name)

    @classmethod
    def import_data(cls,
                    name: str,
                    set_names: list[str],
                    sets_cls: type[DataModel.Sets[T]],
                    importer: Callable[[Path], T | None]) -> Self | None:
        sets_dict: dict[str, T | None] =  {sname: importer(Path('out')/'data'/name/sname)
                                                   for sname in set_names}

        if any(s is None for s in sets_dict.values()):
            logger.error('Could not import data.')
            return None

        logger.info('Imported %s for %s', ', '.join(sets_dict.keys()), name)

        return cls(sets=sets_cls(**sets_dict),
                   name=name)
