from __future__ import annotations

import sys

from qualia_core.datamodel.RawDataModel import RawData, RawDataModel

from .Dataset import Dataset

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class RawDataset(Dataset[RawData]):
    def __init__(self) -> None:
        super().__init__(sets=list(RawDataModel.Sets.fieldnames()))

    @override
    def import_data(self) -> RawDataModel | None:
        return RawDataModel.import_data(name=self.name, sets=self.sets)
