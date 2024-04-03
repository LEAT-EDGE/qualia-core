from __future__ import annotations

import logging
import sys
from typing import Any, Callable

import numpy as np
import numpy.typing

from qualia_core.datamodel import RawDataModel
from qualia_core.datamodel.har.HARDataModel import HARDataModel
from qualia_core.datamodel.har.InfoHAR import Info, Info_dtype
from qualia_core.datamodel.RawDataModel import RawData, RawDataSets
from qualia_core.typing import TYPE_CHECKING

from .Preprocessing import Preprocessing

if TYPE_CHECKING:
    from qualia_core.datamodel.har.Activities import Activities  # noqa: TCH001
    from qualia_core.dataset.Dataset import Dataset  # noqa: TCH001

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)

class DatamodelConverter(Preprocessing[HARDataModel, RawDataModel]):
    @override
    def __call__(self, datamodel: HARDataModel) -> RawDataModel:
        # First generate the class numbers according to the activities present in the dataset
        activitylabels_set: set[Activities] = set()
        for _, s in datamodel.sets:
            for subject in s:
                for activity in subject.activities:
                    activitylabels_set.add(activity.kind)
        activitylabels = sorted(activitylabels_set)
        logger.info('Activity labels: %s', activitylabels)

        sets: dict[str, RawData] = {}
        for name, s in datamodel:
            data_x: list[list[numpy.typing.NDArray[np.float32]]] = []
            data_y: list[int] = []
            infos: list[Info] = []
            for subject in s:
                for i, activity in enumerate(subject.activities):
                    data_x += activity.get_raw_array()
                    #dataY += [int(activity.kind)] * len(activity.samples)
                    data_y += [activitylabels.index(activity.kind)] * len(activity.samples)
                    infos += [Info(subject.name, activity.kind, i)] * len(activity.samples)
            sets[name] = RawData(x=np.array(data_x),
                                 y=np.array(data_y),
                                 info=np.array(infos, dtype=Info_dtype))
        return RawDataModel(sets=RawDataSets(**sets), name=datamodel.name)

    @override
    def import_data(self, dataset: Dataset[Any]) -> Dataset[Any]:
        dataset.import_data = lambda: RawDataModel.import_data(name=dataset.name, sets=dataset.sets)
        return dataset
