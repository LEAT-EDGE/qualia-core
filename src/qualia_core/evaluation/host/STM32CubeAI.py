from pathlib import Path
import re
import sys
from qualia_core.utils.process import subprocesstee                                                                                     
from qualia_core.evaluation.Stats import Stats

import qualia_core.evaluation.STM32CubeAI as STM32CubeAIBase

class STM32CubeAI(STM32CubeAIBase):
    def __init__(self, *args, **kwargs):
        super().__init__(mode='x86')
