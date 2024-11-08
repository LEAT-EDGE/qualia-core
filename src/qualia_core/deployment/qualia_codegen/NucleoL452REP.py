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

class NucleoL452REP(CMake):
    evaluator = QualiaEvaluator # Suggested evaluator

    def __init__(self,
                 projectdir: str | Path | None = None,
                 outdir: str | Path | None = None,
                 core_clock_48mhz: bool = False) -> None:  # noqa: FBT001, FBT002
        super().__init__(projectdir=projectdir if projectdir is not None else
                            resources_to_path(files('qualia_codegen_core.examples'))/'NucleoL452REP',
                         outdir=outdir if outdir is not None else Path('out')/'deploy'/'NucleoL452REP')

        self.__core_clock_48mhz = core_clock_48mhz

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

        if self.__core_clock_48mhz:
            args = (*args, '-D', 'CORE_CLOCK_48MHZ=True')

        return self._run_cmake(args=args, projectdir=self._projectdir, outdir=outdir)

    @override
    def deploy(self, tag: str) -> Deploy | None:
        if not self._run('openocd',
                         '-f', 'interface/stlink.cfg',
                         '-f', 'target/stm32l4x.cfg',
                         '-c', 'init',
                         '-c', 'reset halt; flash write_image erase ./NucleoL452REP; reset; shutdown',
                         cwd=self._outdir/tag):
            return None

        return Deploy(rom_size=self._rom_size(self._outdir/tag/'NucleoL452REP', str(self.__size_bin)),
                      ram_size=self._ram_size(self._outdir/tag/'NucleoL452REP', str(self.__size_bin)),
                      evaluator=self.evaluator)
