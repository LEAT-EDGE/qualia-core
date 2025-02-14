from __future__ import annotations

import sys
from importlib.resources import files
from pathlib import Path

from qualia_core.deployment.Deploy import Deploy
from qualia_core.utils.path import resources_to_path

from .STM32CubeAICMake import STM32CubeAICMake

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class NucleoL452REP(STM32CubeAICMake):
    def __init__(self,
                 projectdir: str | Path | None = None,
                 outdir: str | Path | None = None) -> None:
        super().__init__(projectdir=projectdir if projectdir is not None else
                            resources_to_path(files('qualia_core.assets'))/'projects'/'stm32cubeai'/'NucleoL452REP',
                         outdir=outdir if outdir is not None else Path('out')/'deploy'/'STM32CubeAI-NucleoL452REP')

        self.__size_bin = 'arm-none-eabi-size'

    @override
    def deploy(self, tag: str) -> Deploy | None:
        if not self._run('openocd',
                         '-f', 'interface/stlink.cfg',
                         '-f', 'target/stm32l4x.cfg',
                         '-c', 'init',
                         '-c', 'reset halt; flash write_image erase ./STM32CubeAI-NucleoL452REP; reset; shutdown',
                         cwd=self._outdir/tag):
            return None

        return Deploy(rom_size=self._rom_size(self._outdir/tag/'STM32CubeAI-NucleoL452REP', str(self.__size_bin)),
                      ram_size=self._ram_size(self._outdir/tag/'STM32CubeAI-NucleoL452REP', str(self.__size_bin)),
                      evaluator=self.evaluator)
