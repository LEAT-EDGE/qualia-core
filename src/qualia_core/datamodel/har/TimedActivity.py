from .Activities import Activities
from .Activity import Activity

class TimedActivity(Activity):
    def __init__(self, kind: Activities, timesamples: 'dict[float, TimeSample]'= None):
        self.kind = kind
        self.timesamples = timesamples if timesamples is not None else {}

    @property
    def samples(self):
        keys = list(self.timesamples.keys())
        keys.sort()
        return [self.timesamples[k] for k in keys]

    @samples.setter
    def samples(self, newsamples):
        self.timesamples = {s.t: s for s in newsamples}
