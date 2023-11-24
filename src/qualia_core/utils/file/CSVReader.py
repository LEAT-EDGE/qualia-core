from __future__ import annotations

import csv
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Callable, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

T = TypeVar('T')
U = TypeVar('U')

class CSVReader(Generic[T]):
    @dataclass
    class CSVResult(Generic[U]):
        filename: Path
        content: list[U]

    def read_callback(self,
                      filename: Path,
                      labels: type[T],
                      callback: Callable[[Generator[T, None, None]], None],
                      delimiter: str = ',',
                      skip_header: bool = True) -> None:
        with filename.open(newline='') as f:
            reader = csv.reader(f, delimiter=delimiter)

            if skip_header:
                _ = next(reader)

            callback(labels(*row) for row in reader)

    def read(self,
             filename: Path,
             labels: type[T],
             delimiter: str = ',',
             skip_header: bool = True) -> CSVResult[T]:
        full_content = SimpleNamespace(val=[])

        def callback(content: Generator[T, None, None]) -> None:
            full_content.val = list(content)

        self.read_callback(filename=filename,
                           callback=callback,
                           delimiter=delimiter,
                           labels=labels,
                           skip_header=skip_header)
        return CSVReader.CSVResult(filename=filename, content=full_content.val)
