from __future__ import annotations

import logging
import os
import stat
import sys
from pathlib import Path
from typing import Any

from qualia_core.deployment.toolchain import STM32CubeIDE
from qualia_core.evaluation.target.STM32CubeAI import STM32CubeAI as STM32CubeAIEvaluator
from qualia_core.typing import TYPE_CHECKING

if TYPE_CHECKING:
    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self

    from qualia_core.postprocessing.Converter import Converter  # noqa: TCH001
    from qualia_core.postprocessing.Keras2TFLite import Keras2TFLite  # noqa: TCH001

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)

class STM32CubeAI(STM32CubeIDE):
    evaluator = STM32CubeAIEvaluator # Suggested evaluator

    def __init__(self,
                 projectname: str,
                 projectdir: Path,
                 outdir: Path | None = None) -> None:
        super().__init__(projectname=projectname, projectdir=projectdir, outdir=outdir)

        # Project not compatible with 6.0.0 yet, and 6.0.0 is buggy anyway. Use 5.2.0
        self.__stm32cubeai_bin = Path.home()/'STM32Cube'/'Repository'/'Packs'/'STMicroelectronics'/'X-CUBE-AI'/'5.2.0'/'Utilities'/'linux'/'stm32ai'
        #self.__stm32cubeai_bin = next((Path.home()/'STM32Cube'/'Repository'/'Packs'/'STMicroelectronics'/'X-CUBE-AI').glob('*'))/'Utilities'/'linux'/'stm32ai'

    def _create_modeloutdir(self, modelpath: Path) -> None:
        modelpath.mkdir(parents=True, exist_ok=True)

    def __write_model(self, model: Keras2TFLite, modelpath: Path) -> None:
        with modelpath.open('wb') as f:
            _ = f.write(model.data)

    def __generate(self, modelpath: Path, compression: int) -> bool:
        if compression != 1:
            logger.error('FIXME: Evaluation not logging compression level != 1')
            raise ValueError
        if compression not in [1, 4, 8]:
            logger.error('Compression factor %s is not supported, must be either 1, 4 or 8', compression)
            raise ValueError
        # Sometimes the executable may not have have exec permission, set it
        if not os.access(self.__stm32cubeai_bin, os.X_OK):
            self.__stm32cubeai_bin.chmod(self.__stm32cubeai_bin.stat().st_mode | stat.S_IXUSR)

        return self._run(self.__stm32cubeai_bin,
                            'generate',
                            '--model', str(modelpath),
                            '--output', str(self._projectdir/'X-CUBE-AI'/'App'),
                            '--compression', str(compression),
                            '--workspace', str(self._workspacedir),
                        )

    @override
    def prepare(self,
                tag: str,
                model: Converter[Any],
                optimize: str,
                compression: int) -> Self | None:
        # Keep here for isinstance() to avoid circual import
        from qualia_core.postprocessing.Keras2TFLite import Keras2TFLite

        if optimize != 'cmsis-nn':
            logger.error('cmsis-nn optimize mandatory for %s', type(self).__name__)
            raise ValueError

        if not isinstance(model, Keras2TFLite):
            logger.error('%s excepts the model to come from a Keras2TFLite Converter', type(self).__name__)
            raise TypeError


        modeloutdir = Path('out')/'deploy'/'stm32cubeai'
        modelpath = modeloutdir/f'{tag}.tflite'
        self._create_modeloutdir(modeloutdir)
        self._create_outdir()
        self._clean_workspace()
        self.__write_model(model=model, modelpath=modelpath)

        if not self.__generate(modelpath=modelpath, compression=compression):
            return None
        if not self._build():
            return None
        self._copy_buildproduct(tag=tag)
        return self
