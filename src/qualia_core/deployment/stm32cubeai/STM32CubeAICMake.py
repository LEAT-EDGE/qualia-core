from __future__ import annotations

import logging
import os
import stat
import sys
from pathlib import Path
from typing import Any

from qualia_core.deployment.toolchain.CMake import CMake
from qualia_core.evaluation.target.STM32CubeAI import STM32CubeAI as STM32CubeAIEvaluator
from qualia_core.typing import TYPE_CHECKING

if TYPE_CHECKING:
    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self

    from qualia_core.postprocessing.Converter import Converter  # noqa: TC001
    from qualia_core.postprocessing.Keras2TFLite import Keras2TFLite  # noqa: TC001

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)

class STM32CubeAICMake(CMake):
    evaluator = STM32CubeAIEvaluator # Suggested evaluator

    def __init__(self,
                 projectdir: str | Path,
                 outdir: str | Path) -> None:
        super().__init__(projectdir=projectdir, outdir=outdir)

        #  Built-in project made for 8.1.0
        self.__stm32cubeai_bin = Path.home()/'STM32Cube'/'Repository'/'Packs'/'STMicroelectronics'/'X-CUBE-AI'/'8.1.0'/'Utilities'/'linux'/'stm32ai'
        #self.__stm32cubeai_bin = next((Path.home()/'STM32Cube'/'Repository'/'Packs'/'STMicroelectronics'/'X-CUBE-AI').glob('*'))/'Utilities'/'linux'/'stm32ai'

    def _create_modeloutdir(self, modelpath: Path) -> None:
        modelpath.mkdir(parents=True, exist_ok=True)

    def __write_model(self, model: Keras2TFLite, modelpath: Path) -> None:
        with modelpath.open('wb') as f:
            _ = f.write(model.data)

    def __generate(self, modelpath: Path, modeloutdir: Path, compression: int) -> bool:
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
                            '--output', str(modeloutdir),
                            '--compression', str(compression),
                        )

    @override
    def _build(self,
               modeldir: Path,
               optimize: str,
               outdir: Path) -> bool:
        args = ('-D', f'NETWORK_DIR={modeldir.resolve()!s}')

        return self._run_cmake(args=args, projectdir=self._projectdir, outdir=outdir)

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


        outdir = self._outdir / tag
        modeloutdir = Path('out')/'deploy'/'stm32cubeai'
        modelpath = modeloutdir/f'{tag}.tflite'
        self._create_modeloutdir(modeloutdir)
        self._create_outdir(outdir)
        self.__write_model(model=model, modelpath=modelpath)

        if not self.__generate(modelpath=modelpath, modeloutdir=modeloutdir, compression=compression):
            return None

        if not self._build(modeldir=modeloutdir, optimize=optimize, outdir=outdir):
            return None

        return self
