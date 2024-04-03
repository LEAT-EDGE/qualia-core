
from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Any

from qualia_core.datamodel.RawDataModel import RawDataModel

from .Preprocessing import Preprocessing

if TYPE_CHECKING:
    from qualia_core.dataset.Dataset import Dataset

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)

class DatasetSplitterBySubjects(Preprocessing[RawDataModel, RawDataModel]):
    """Warning: must be applied after Window to get correct split and randomization of windows."""

    def __init__(self,
                 source_subjects: list[Any],
                 dest_subjects: list[Any],
                 source: str = 'train',
                 dest: str = 'test') -> None:
        super().__init__()
        self.__source_subjects = source_subjects
        self.__dest_subjects = dest_subjects
        self.__source = source
        self.__dest = dest

    @override
    def __call__(self, datamodel: RawDataModel) -> RawDataModel:
        source = getattr(datamodel.sets, self.__source)
        dest = getattr(datamodel.sets, self.__dest)

        dest = [s for s in source if s.name in self.__dest_subjects]
        source = [s for s in source if s.name in self.__source_subjects]

        setattr(datamodel.sets, self.__dest, dest)
        setattr(datamodel.sets, self.__source, source)

        return datamodel

    @override
    def import_data(self, dataset: Dataset[Any]) -> Dataset[Any]:
        # Add dest to list of sets for dataset being loaded
        if self.__dest not in dataset.sets:
            dataset.sets.append(self.__dest)
        return dataset
