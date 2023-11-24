from pathlib import Path
import shutil
from collections import namedtuple
from qualia_core.utils.process import subprocesstee

class Linux:
    import qualia_core.evaluation.host.STM32CubeAI as evaluator # Suggested evaluator

    def prepare(self, tag, model, optimize: str, compression: int):
        raise NotImplemented('Should write TFLite model')
        print(self.__class__.__name__, 'Info: running locally, nothing to prepare')
        return self

    def deploy(self, tag):
        print(self.__class__.__name__, 'Info: running locally, nothing to deploy')

        return namedtuple('Deploy', ['rom_size', 'evaluator'])(self.__rom_size(tag), self.evaluator)

    def __rom_size(self, tag):
        return -1

