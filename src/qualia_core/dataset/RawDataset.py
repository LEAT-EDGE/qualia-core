from __future__ import annotations

import sys

from qualia_core.datamodel.RawDataModel import RawData, RawDataChunks, RawDataChunksModel, RawDataModel

from .Dataset import Dataset

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override


class RawDataset(Dataset[RawData]):
    def __init__(self) -> None:
        super().__init__(sets=list(RawDataModel.Sets.fieldnames()))

    @override
    def import_data(self) -> RawDataModel:
        rdm = RawDataModel(name=self.name)
        return rdm.import_sets(set_names=self.sets)


class RawDatasetChunks(Dataset[RawDataChunks, RawData]):
    def __init__(self) -> None:
        super().__init__(sets=list(RawDataChunksModel.Sets.fieldnames()))

    @override
    def import_data(self) -> RawDataModel:
        rdm = RawDataModel(name=self.name)
        return rdm.import_sets(set_names=self.sets, importer=RawDataChunks.import_data)
