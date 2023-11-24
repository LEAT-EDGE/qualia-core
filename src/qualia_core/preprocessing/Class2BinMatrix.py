import numpy as np

from .Preprocessing import Preprocessing

class Class2BinMatrix(Preprocessing):
    '''Warning: must be applied after Window'''
    def __init__(self, classes: int=None):
        self.__classes = classes

    def __call__(self, datamodel):
        for _, s in datamodel:
            if len(s.y.shape) != 1:
                raise ValueError(f'Unsupported dimensions for {self.__class__.__name__}: {len(s.y.shape)}')
            if len(s.y) <= 0: # Handle empty sets
                continue

            if not self.__classes:
                s.y = np.eye(np.max(s.y) + 1)[s.y]
            else:
                s.y = np.eye(self.__classes)[s.y]

        return datamodel
