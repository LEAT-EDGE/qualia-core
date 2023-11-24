from .Preprocessing import Preprocessing

class RemoveSensor(Preprocessing):
    def __init__(self, sensorkinds, noconvert: bool=False):
        import qualia_core.datamodel.sensor as sensors
        for i, sensorkind in enumerate(sensorkinds):
            if not noconvert and isinstance(sensorkind, str):
                sensorkinds[i] = getattr(sensors, sensorkind)
        self.__excludesensors = sensorkinds

    def __call__(self, datamodel):
        for _, subjectset in datamodel:
            for subject in subjectset:
                for activity in subject.activities:
                    for sample in activity.samples:
                        sample.sensors = [sensor for sensor in sample.sensors if sensor.__class__ not in self.__excludesensors]
                    activity.samples = [sample for sample in activity.samples if len(sample.sensors) > 0] # Remove empty samples
        return datamodel
