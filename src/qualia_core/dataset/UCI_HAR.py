import logging
import sys
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing

from qualia_core.datamodel import FeatureSample, WindowedSample
from qualia_core.datamodel.har import Activities, Activity, HARDataModel, Subject

from .HARDataset import HARDataset

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)

class UCI_HAR(HARDataset):
    activitylist: Final[dict[str, Activities]] = {
            '1': Activities.WALKING,
            '2': Activities.WALKING_UPSTAIRS,
            '3': Activities.WALKING_DOWNSTAIRS,
            '4': Activities.SITTING,
            '5': Activities.STANDING,
            '6': Activities.LYING,
            }

    def __init__(self, path: str='', variant: str='features') -> None:
        super().__init__()
        self.__path = Path(path)
        self.__variant = variant
        self.sets.remove('valid')

    def __load_features_data(self, part: str, path: Path) -> list[Subject]:
        # load features, labels and subjects
        with (path/part/f'X_{part}.txt').open('r') as fxtrain, \
             (path/part/f'y_{part}.txt').open('r') as fytrain, \
             (path/part/f'subject_{part}.txt').open('r') as fsubject:

             sxtrain = fxtrain.read().splitlines()
             sytrain = fytrain.read().splitlines()
             ssubject = fsubject.read().splitlines()

        # populate datamodel
        subjects: list[Subject] = []
        activities: list[Activity] = []
        samples: list[FeatureSample] = []
        lastact = sytrain[0]
        lastsub = ssubject[0]

        for data, label, subject in zip(sxtrain, sytrain, ssubject):
            if lastact != label or lastsub != subject:
                activities.append(Activity(UCI_HAR.activitylist[lastact], samples))
                samples = []
                lastact = label

            if lastsub != subject:
                subjects.append(Subject(subject, activities))
                activities = []
                lastsub = subject

            samples.append(FeatureSample([float(d) for d in data.split()]))

        activities.append(Activity(UCI_HAR.activitylist[label], samples))

        subjects.append(Subject(subject, activities, part=part))

        return subjects

    def __load_features(self, path: Path) -> HARDataModel:
        return HARDataModel(sets=HARDataModel.Sets(self.__load_features_data('train', path=path),
                                                   self.__load_features_data('test', path=path)),
                            name=self.name)

    def __load_raw_data(self, part: str, path: Path) -> list[Subject]:
        filenames = [f'{sensorname}_{dim}_{part}.txt'
                        for sensorname in ('body_acc', 'body_gyro', 'total_acc')
                            for dim in ('x', 'y', 'z')]

        # Load body_acc,body_gyro,total_acc x,y,z data and concatenate to create (,128,7)
        xtrain_list: list[numpy.typing.NDArray[np.float32]] = []
        for fn in filenames:
            with (path/part/'Inertial Signals'/fn).open() as f:
                xtrain_list.append(np.array([[float(d) for d in line.rstrip().split()] for line in f.readlines()],
                                             dtype=np.float32))
        xtrain = np.dstack(xtrain_list)

        # load labels and subjects
        with (path/part/f'y_{part}.txt').open('r') as fytrain, \
             (path/part/f'subject_{part}.txt').open('r') as fsubject:

            sytrain = fytrain.read().splitlines()
            ssubject = fsubject.read().splitlines()

        # populate datamodel
        subjects: list[Subject] = []
        activities: list[Activity] = []
        samples: list[WindowedSample] = []
        lastact = sytrain[0]
        lastsub = ssubject[0]


        for data, label, subject in zip(xtrain, sytrain, ssubject):
            if lastact != label or lastsub != subject:
                activities.append(Activity(int(float(lastact))-1, samples))
                samples = []
                lastact = label

            if lastsub != subject:
                subjects.append(Subject(lastsub, activities))
                activities = []
                lastsub = subject
            samples.append(WindowedSample(data))

        activities.append(Activity(int(float(lastact))-1, samples))

        subjects.append(Subject(lastsub, activities, part=part))

        return subjects

    def __load_raw(self, path: Path) -> HARDataModel:
        return HARDataModel(sets=HARDataModel.Sets(train=self.__load_raw_data('train', path=path),
                                                   test=self.__load_raw_data('test', path=path)),
                            name=self.name)

    @override
    def __call__(self) -> HARDataModel:
        if self.__variant == 'features':
            return self.__load_features(self.__path)

        if self.__variant == 'raw':
            return self.__load_raw(self.__path)

        logger.error('Only features and raw variants supported')
        raise ValueError

    @property
    @override
    def name(self) -> str:
        return f'{super().name}_{self.__variant}'
