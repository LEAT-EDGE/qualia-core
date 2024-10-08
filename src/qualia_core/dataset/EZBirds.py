from __future__ import annotations

import json
import logging
import sys

from qualia_core.typing import TYPE_CHECKING

from .GSC import GSC

if TYPE_CHECKING:
    from pathlib import Path  # noqa: TCH003

    from qualia_core.datamodel import RawDataModel  # noqa: TCH001

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)

class EZBirds(GSC):
    def __init__(self, path: str) -> None:
        super().__init__(path=path,
                         variant='',
                         subset='',
                         train_valid_split=True,
                         record_length=48000)

    def __load_labels(self, path: Path) -> dict[str, int | None]:
        with (path / 'labels.json').open('r') as f:
            labels = json.load(f)
        return {label: i for i, label in enumerate(labels)}

    @override
    def __call__(self) -> RawDataModel:
        class_list = self.__load_labels(self._path)

        return self._load_v2(self._path, class_list=class_list)

    @property
    @override
    def name(self) -> str:
        return f'{super().name}'
