from pathlib import Path
from collections import namedtuple

class Linux:
    import qualia_core.evaluation.host.TFLite as evaluator # Suggested evaluator

    def __create_outdir(self, modeloutdir):
        modeloutdir.mkdir(parents=True, exist_ok=True)

    def __write_model(self, model, modelpath):
        with modelpath.open('wb') as f:
            f.write(model.data)

    def prepare(self, tag, model, optimize: str, compression: int):
        modeloutdir = Path('out')/'deploy'/'tflite'
        modelpath = modeloutdir/f'{tag}.tflite'
        self.__create_outdir(modeloutdir)
        self.__write_model(model=model, modelpath=modelpath)

        return self

    def deploy(self, tag):
        print(self.__class__.__name__, 'Info: running locally, nothing to deploy')

        return namedtuple('Deploy', ['rom_size', 'evaluator'])(self.__rom_size(tag), self.evaluator)

    def __rom_size(self, tag):
        return -1
