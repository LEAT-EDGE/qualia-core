from pathlib import Path
import shutil
import os
import stat
from qualia_core.utils.process import subprocesstee

from qualia_core.deployment.toolchain import STM32CubeIDE

class STM32CubeAI(STM32CubeIDE):
    import qualia_core.evaluation.target.STM32CubeAI as evaluator # Suggested evaluator

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Project not compatible with 6.0.0 yet, and 6.0.0 is buggy anyway. Use 5.2.0
        self.__stm32cubeai_bin = Path.home()/'STM32Cube'/'Repository'/'Packs'/'STMicroelectronics'/'X-CUBE-AI'/'5.2.0'/'Utilities'/'linux'/'stm32ai'
        #self.__stm32cubeai_bin = next((Path.home()/'STM32Cube'/'Repository'/'Packs'/'STMicroelectronics'/'X-CUBE-AI').glob('*'))/'Utilities'/'linux'/'stm32ai'

    def _create_outdir(self, modelpath):
        super()._create_outdir()
        modelpath.mkdir(parents=True, exist_ok=True)

    def __write_model(self, model, modelpath):
        with modelpath.open('wb') as f:
            f.write(model.data)

    def __generate(self, modelpath: Path, compression: int):
        if compression != 1:
            raise ValueError(f'FIXME: Evaluation not logging compression level != 1')
        if compression not in [1, 4, 8]:
            raise ValueError(f'Compression factor {compression} is not supported, must be either 1, 4 or 8')
        # Sometimes the executable may not have have exec permission, set it
        if not os.access(self.__stm32cubeai_bin, os.X_OK):
            os.chmod(self.__stm32cubeai_bin,  os.stat(self.__stm32cubeai_bin).st_mode | stat.S_IXUSR)

        return self._run(self.__stm32cubeai_bin,
                            'generate',
                            '--model', str(modelpath),
                            '--output', str(self._projectdir/'X-CUBE-AI'/'App'),
                            '--compression', str(compression),
                            '--workspace', str(self._workspacedir)
                        )

    def prepare(self, tag, model, optimize: str, compression: int):
        if optimize != 'cmsis-nn':
            raise ValueError(f'cmsis-nn optimize mandatory for {self.__class__.__name__}')

        modeloutdir = Path('out')/'deploy'/'stm32cubeai'
        modelpath = modeloutdir/f'{tag}.tflite'
        self._create_outdir(modeloutdir)
        self._clean_workspace()
        self.__write_model(model=model, modelpath=modelpath)

        if not self.__generate(modelpath=modelpath, compression=compression):
            return None
        if not self._build(tag=tag):
            return None
        self._copy_buildproduct(tag=tag)
        return self
