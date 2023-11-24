import numpy as np

from .Preprocessing import Preprocessing

class BandPassFilter(Preprocessing):
    '''Band pass filter
       Requires fixed sampling frequency, do an interpolation before.
    '''
    def __init__(self, f1: float, f2: float, fs: float, sensorkind, dimension: str):
        self.__f1 = f1
        self.__f2 = f2
        self.__fs = fs
        self.__dimension = dimension

        if isinstance(sensorkind, str):
            import qualia_core.datamodel.sensor as sensors
            sensorkind = getattr(sensors, sensorkind)
        self.__sensorkind = sensorkind

    def __filter(self, array):
        import scipy.signal

        nyq = 0.5 * self.__fs
        w1 = self.__f1 / nyq
        w2 = self.__f2 / nyq
        b, a = scipy.signal.butter(1, [w1, w2], btype='bandpass')
        if len(array) < len(a) * len(b): # Do not apply filter if data smaller than filter
            print(f'Warning: band pass filter not applied for current activity. Signal len {len(array)} smaller than filter length {len(a) * len(b)}')
            return array
        else:
            return scipy.signal.filtfilt(b, a, array)

    def __call__(self, datamodel):

        for name, subjectset in datamodel:
            for subject in subjectset:
                for activity in subject.activities:
                    signal = np.array([getattr(sensor, self.__dimension) for sample in activity.samples
                                                for sensor in sample.sensors if isinstance(sensor, self.__sensorkind)])
                    filtered = self.__filter(signal)
                    ifiltered = np.nditer(filtered, order='A') # using default 'K' order will reverse element, probably a side-effect
                                                               # of filtfilt

                    for sample in activity.samples:
                        for sensor in sample.sensors:
                            if not isinstance(sensor, self.__sensorkind):
                                continue
                            setattr(sensor, self.__dimension, next(ifiltered))
        return datamodel
