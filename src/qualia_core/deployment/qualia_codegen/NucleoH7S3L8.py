from __future__ import annotations

import logging
import sys
from importlib.resources import files
from pathlib import Path

from qualia_core.deployment.Deploy import Deploy
from qualia_core.evaluation.target.Qualia import Qualia as QualiaEvaluator
from qualia_core.utils.path import resources_to_path

from .CMake import CMake

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)

class NucleoH7S3L8(CMake):
    evaluator = QualiaEvaluator # Suggested evaluator

    def __init__(self,
                 projectdir: str | Path | None = None,
                 outdir: str | Path | None = None) -> None:
        super().__init__(projectdir=projectdir if projectdir is not None else
                            resources_to_path(files('qualia_codegen_core.examples'))/'NucleoH7S3L8',
                         outdir=outdir if outdir is not None else Path('out')/'deploy'/'NucleoH7S3L8')

        self.__size_bin = 'arm-none-eabi-size'

    @override
    def _validate_optimize(self, optimize: str) -> None:
        if optimize and optimize != 'cmsis-nn':
            logger.error('Optimization %s not available for %s', optimize, type(self).__name__)
            raise ValueError

    @override
    def _build(self,
               modeldir: Path,
               optimize: str,
               outdir: Path) -> bool:
        args = ('-D', f'MODEL_DIR={modeldir.resolve()!s}')
        if optimize == 'cmsis-nn':
            args = (*args, '-D', 'WITH_CMSIS_NN=True')

        return self._run_cmake(args=args, projectdir=self._projectdir, outdir=outdir)

    @override
    def deploy(self, tag: str) -> Deploy | None:
        # if not self._run('openocd',
        #                  '-f', 'interface/stlink.cfg',
        #                  '-f', 'target/stm32h7x.cfg',
        #                  '-c', 'init',
        #                  '-c', 'reset halt; flash write_image erase ./NucleoH7S3L8; reset; shutdown',
        #                  cwd=self._outdir/tag):
        elf = self._outdir/tag/'NucleoH7S3L8'
        elf = elf.rename(elf.with_suffix('.elf')) if elf.exists() else elf.with_suffix('.elf')

        if not self._run('STM32_Programmer_CLI',
                         '--connect', 'port=SWD', 'mode=UR', 'reset=hwRst',
                         '--download', str(elf), '0x08000000',
                         '--verify',
                         '-hardRst'):
            return None

        return Deploy(rom_size=self._rom_size(elf, str(self.__size_bin)),
                      ram_size=self._ram_size(elf, str(self.__size_bin)),
                      evaluator=self.evaluator)
