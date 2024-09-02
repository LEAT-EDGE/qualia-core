import importlib.util
import logging

from .BandPassFilter import BandPassFilter
from .Class2BinMatrix import Class2BinMatrix
from .CopySet import CopySet
from .DatamodelConverter import DatamodelConverter
from .DatasetSplitter import DatasetSplitter
from .DatasetSplitterBySubjects import DatasetSplitterBySubjects
from .MFCC import MFCC
from .Normalize import Normalize
from .PrintHARDataModelSummary import PrintHARDataModelSummary
from .RemoveActivity import RemoveActivity
from .RemoveSensor import RemoveSensor
from .Reshape2DTo1D import Reshape2DTo1D
from .Window import Window

__all__ = ['BandPassFilter',
           'Class2BinMatrix',
           'CopySet',
           'DatamodelConverter',
           'DatasetSplitterBySubjects',
           'DatasetSplitter',
           'MFCC',
           'Normalize',
           'PrintHARDataModelSummary',
           'RemoveActivity',
           'RemoveActivity',
           'RemoveSensor',
           'Reshape2DTo1D',
           'Window']

logger = logging.getLogger(__name__)

if importlib.util.find_spec('matplotlib') is None:
    logger.warning('Matplotlib is required for VisualizeActivities, VisualizeWindows')
else:
    from .VisualizeActivities import VisualizeActivities
    from .VisualizeWindows import VisualizeWindows

    __all__ += ['VisualizeActivities',
                'VisualizeWindows']
