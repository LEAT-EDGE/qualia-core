from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from qualia_core.datamodel import TimeSample
from qualia_core.datamodel.har import Activities, Activity, HARDataModel, Subject
from qualia_core.datamodel.sensor import Accelerometer, Barometer, Gyroscope
from qualia_core.utils.file import CSVReader, DirectoryReader

from .HARDataset import HARDataset

if TYPE_CHECKING:
    from qualia_core.datamodel.sensor.Sensor import Sensor

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class EllcieHAR(HARDataset):
    @dataclass
    class Row:
        T: str
        Ax: str
        Ay: str
        Az: str
        Gx: str
        Gy: str
        Gz: str
        P: str
        CLASS: str


    activitylist: Final[dict[str, Activities]] = {
        'STANDING': Activities.STANDING,
        'STAND_TO_SIT': Activities.STAND_TO_SIT,
        'SITTING': Activities.SITTING,
        'SIT_TO_STAND': Activities.SIT_TO_STAND,
        'WALKING': Activities.WALKING,
        'SIT_TO_LIE': Activities.SIT_TO_LIE,
        'LYING': Activities.LYING,
        'LIE_TO_SIT': Activities.LIE_TO_SIT,
        'WALKING_UPSTAIRS': Activities.WALKING_UPSTAIRS,
        'WALKING_DOWNSTAIRS': Activities.WALKING_DOWNSTAIRS,
        'RUNNING': Activities.RUNNING,
        'DRINKING': Activities.DRINKING,
        'DRIVING': Activities.DRIVING,
    }

    def __init__(self,
                 path: str = '',
                 variant: str = '1',
                 files: list[str] | None = None) -> None:
        super().__init__()
        self.__dr = DirectoryReader()
        self.__cr: CSVReader[EllcieHAR.Row] = CSVReader()
        self.__path = Path(path)
        self.__variant = variant
        self.__files = files

        if self.__variant not in ['PACK-2', 'UCA-EHAR']:
            raise ValueError('Unsupported variant ' + variant)

        self.sets.remove('valid')

    @override
    def __call__(self) -> HARDataModel:
        files = self.__dr.read(self.__path, ext='.csv')
        filesd = [self.__cr.read(f, delimiter=';', labels=EllcieHAR.Row) for f in files]

        subjects: dict[str, Subject] = {}

        for fd in filesd:
            name = fd.filename
            if self.__files is not None and not any(s.lower() in name.name.lower() for s in self.__files):
                continue

            data = fd.content
            subject_name = name.stem
            if self.__variant == 'PACK-2':
                subject_name = subject_name.split('_')[-2]
            elif self.__variant == 'UCA-EHAR':
                subject_name = subject_name.split('_')[1]

            if subject_name not in subjects:
                subjects[subject_name] = Subject(subject_name, [])
            activities = subjects.setdefault(subject_name, Subject(subject_name, [])).activities

            lastact = data[0].CLASS
            firstactt = float(data[0].T)
            samples: list[TimeSample] = []

            for d in data:
                if lastact != d.CLASS:
                    if lastact != '': # Drop unlabelled samples
                        activities.append(Activity(self.activitylist[lastact], samples))
                    samples = []
                    lastact = d.CLASS
                    firstactt = float(d.T)

                absolutet = float(d.T)

                sensors: list[Sensor] = []
                if d.Ax != '': # acc/gyro should be synced
                    sensors.append(Accelerometer(float(d.Ax), float(d.Ay), float(d.Az)))
                    sensors.append(Gyroscope(float(d.Gx), float(d.Gy), float(d.Gz)))
                if d.P != '':
                    sensors.append(Barometer(float(d.P)))
                sample = TimeSample(t=absolutet - firstactt, sensors=sensors)
                samples.append(sample)

            if d.CLASS != '': # Drop unlabelled samples
                activities.append(Activity(self.activitylist[d.CLASS], samples))

        return HARDataModel(sets=HARDataModel.Sets(train=list(subjects.values())), name=self.name)

    @property
    @override
    def name(self) -> str:
        return f'{super().name}_{self.__variant}'
