import numpy as np

from .Preprocessing import Preprocessing

class DatasetSplitterBySubjects(Preprocessing):
    '''Warning: must be applied after Window to get correct split and randomization of windows'''
    def __init__(self, source_subjects: list, dest_subjects: list, source: str='train', dest: str='test'):
        self.__source_subjects = source_subjects
        self.__dest_subjects = dest_subjects
        self.__source = source
        self.__dest = dest

    def __call__(self, datamodel):
        """
            Splits Dataset in two.
            Set dest with 'ratio' percentage of vectors taken from source and set source with '1-ratio' percentage of vectors
            taken from source.
        """
        source = getattr(datamodel.sets, self.__source)
        dest = getattr(datamodel.sets, self.__dest)

        dest = [s for s in source if s.name in self.__dest_subjects]
        source = [s for s in source if s.name in self.__source_subjects]

        setattr(datamodel.sets, self.__dest, dest)
        setattr(datamodel.sets, self.__source, source)

        return datamodel
