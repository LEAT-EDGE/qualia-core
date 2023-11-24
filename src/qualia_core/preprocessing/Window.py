import numpy as np

from .Preprocessing import Preprocessing

class Window(Preprocessing):
    '''Warning: Must be applied before Class2BinMatrix and DatasetSplitter'''
    def __init__(self, size: int=0, stride: int=1, no_overlapping_labels: bool=True, unique_label_per_window: bool=True):
        self.__size = size
        self.__stride = stride
        self.__no_overlapping_labels = no_overlapping_labels
        self.__unique_label_per_window = unique_label_per_window

    # Inspired from https://stackoverflow.com/a/45730836 with support added for multidimensional arrays, windowing in 1st dim
    def __window(self, a, size=4, stride=2):
        shape = (a.shape[0] - size + 1, size) + a.shape[1:]
        strides = a.strides[:1] * 2 + a.strides[1:]
        view = np.lib.stride_tricks.as_strided(a, strides=strides, shape=shape)
        view = view[0::stride]
        return view.copy()

    def __call__(self, datamodel):
        for name, s in datamodel:
            if len(s.data) < 1: # Ignore empty sets
                continue
            s.data = self.__window(s.data, size=self.__size, stride=self.__stride)
            s.labels = self.__window(s.labels, size=self.__size, stride=self.__stride)
            s.info = self.__window(s.info, size=self.__size, stride=self.__stride)

            if self.__no_overlapping_labels:
                is_not_overlapping = np.all(s.labels == s.labels[:,0].reshape((-1, 1)), axis=-1)
                non_overlapping_i = np.where(is_not_overlapping)

                s.data = s.data[non_overlapping_i]
                s.labels = s.labels[non_overlapping_i]
                s.info = s.info[non_overlapping_i]

            print(f'Number of vectors in {name} set: {len(s.data)}')

            if self.__unique_label_per_window:
                s.labels = np.array([np.bincount(w).argmax() for w in s.y])

                print(f'Number of vector per class in {name} set: {np.bincount(s.labels)}')
                print(f'Balancing weights: {np.max(np.bincount(s.labels)) / np.bincount(s.labels)}')

        return datamodel
