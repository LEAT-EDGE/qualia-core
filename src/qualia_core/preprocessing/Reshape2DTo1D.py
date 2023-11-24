import numpy as np

from .Preprocessing import Preprocessing

class Reshape2DTo1D(Preprocessing):
    def __call__(self, datamodel):
        import torch
        for name, s in datamodel:
            s.x = s.x.reshape((s.x.shape[0], s.x.shape[1] * s.x.shape[2], s.x.shape[3]))
            print(s.x.shape)

        return datamodel
