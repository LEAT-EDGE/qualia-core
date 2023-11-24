from collections import namedtuple
import numpy as np

Info = namedtuple('Info', ['subject', 'activity', 'activity_i'])
Info_dtype = [('subject', 'U16'), ('activity', 'U32'), ('activity_i', int)]
