from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Any

from qualia_core import random
from qualia_core.datamodel.RawDataModel import RawData, RawDataModel

from .Preprocessing import Preprocessing

if TYPE_CHECKING:
    from qualia_core.dataset.Dataset import Dataset

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)

class DatasetSplitter(Preprocessing[RawDataModel, RawDataModel]):
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
        """Split Dataset in two.

        Set dest with 'ratio' percentage of vectors taken from source and set source
        with '1-ratio' percentage of vectors taken from source.
        """
        source: RawData | None = getattr(datamodel.sets, self.__source)
        dest: RawData | None = getattr(datamodel.sets, self.__dest)
        if source is None:
            logger.error('Source set %s does not exist in dataset', self.__source)
            raise ValueError

        perms = random.shared.generator.permutation(len(source.x))
        destperms = perms[0:int(len(perms) * self.__ratio)]
        sourceperms = perms[len(destperms):]

        dest_x = source.x[destperms]
        dest_y = source.y[destperms]
        dest_info = source.info[sourceperms] if source.info is not None else None

        if dest is None:
            # Destination set does not exist, create it.
            dest = RawData(x=dest_x, y=dest_y, info=dest_info)
            setattr(datamodel.sets, self.__dest, dest)
        else:
            dest.x = dest_x
            dest.y = dest_y
            dest.info = dest_info

        source.x = source.x[sourceperms]
        source.y = source.y[sourceperms]
        if source.info is not None:
            source.info = source.info[sourceperms]

        return datamodel

    @override
    def import_data(self, dataset: Dataset[Any]) -> Dataset[Any]:
        # Add dest to list of sets for dataset being loaded
        if self.__dest not in dataset.sets:
            dataset.sets.append(self.__dest)
        return dataset
