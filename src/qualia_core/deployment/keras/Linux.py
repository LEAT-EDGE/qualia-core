from pathlib import Path
from collections import namedtuple

class Linux:
    import qualia_core.evaluation.host.Keras as evaluator # Suggested evaluator

    def prepare(self, tag, model, optimize: str, compression: int):
        from qualia_core.learningframework import Keras

        framework = Keras()
        framework.export(model, tag, path=Path('out')/'deploy'/'keras')

        return self

    def deploy(self, tag):
        print(self.__class__.__name__, 'Info: running locally, nothing to deploy')

        return namedtuple('Deploy', ['rom_size', 'evaluator'])(self.__rom_size(tag), self.evaluator)

    def __rom_size(self, tag):
        return -1
