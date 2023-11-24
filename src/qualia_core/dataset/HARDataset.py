from __future__ import annotations

import sys

from qualia_core.datamodel.har.HARDataModel import HARDataModel
from qualia_core.datamodel.har.Subject import Subject

from .Dataset import Dataset

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class HARDataset(Dataset[list[Subject]]):
    def __init__(self) -> None:
        super().__init__(sets=list(HARDataModel.Sets.fieldnames()))

    @override
    def import_data(self) -> HARDataModel | None:
        raise NotImplementedError
