from __future__ import annotations

import copy
import logging
import sys
from typing import TYPE_CHECKING, Any

from qualia_core.datamodel.RawDataModel import RawData, RawDataModel

from .Preprocessing import Preprocessing

if TYPE_CHECKING:
    from qualia_core.dataset.Dataset import Dataset

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)

class CopySet(Preprocessing[RawDataModel, RawDataModel]):
    def __init__(self,
                 source: str = 'train',
                 dest: str = 'test',
                 ratio: float = 0.1) -> None:
        super().__init__()
        self.__source = source
        self.__dest = dest
        self.__ratio = ratio

    @override
    def __call__(self, datamodel: RawDataModel) -> RawDataModel:
        """Copy source set to destination set, e.g., test to valid."""
        source: RawData | None = getattr(datamodel.sets, self.__source)
        dest: RawData | None = getattr(datamodel.sets, self.__dest)
        if source is None:
            logger.error('Source set %s does not exist in dataset', self.__source)
            raise ValueError

        dest_x = copy.deepcopy(source.x)
        dest_y = copy.deepcopy(source.y)
        dest_info = copy.deepcopy(source.info)

        if dest is None:
            # Destination set does not exist, create it.
            dest = RawData(x=dest_x, y=dest_y, info=dest_info)
            setattr(datamodel.sets, self.__dest, dest)
        else:
            dest.x = dest_x
            dest.y = dest_y
            dest.info = dest_info

        return datamodel

    @override
    def import_data(self, dataset: Dataset[Any]) -> Dataset[Any]:
        # Add dest to list of sets for dataset being loaded
        if self.__dest not in dataset.sets:
            dataset.sets.append(self.__dest)
        return dataset
