from __future__ import annotations

import logging
import sys
from importlib.resources import files
from pathlib import Path

from qualia_core.deployment.Deploy import Deploy
from qualia_core.evaluation.target.Qualia import Qualia as QualiaEvaluator
from qualia_core.utils.path import resources_to_path

from .NucleoL452REP import NucleoL452REP

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)


class NucleoH7A3ZIQ(NucleoL452REP):
    evaluator = QualiaEvaluator  # Suggested evaluator

    def __init__(self,
                 projectdir: str | Path | None = None,
                 outdir: str | Path | None = None) -> None:
        super().__init__(projectdir=projectdir if projectdir is not None else
                            resources_to_path(files('qualia_codegen_core.examples'))/'NucleoH7A3ZIQ',
                         outdir=outdir if outdir is not None else Path('out')/'deploy'/'NucleoH7A3ZIQ')

    @override
    def deploy(self, tag: str) -> Deploy | None:
        if not self._run('openocd',
                         '-f', 'interface/stlink.cfg',
                         '-f', 'target/stm32h7x.cfg',
                         '-c', 'init',
                         '-c', 'reset halt; flash write_image erase ./NucleoH7A3ZIQ.elf; reset; shutdown',
                         cwd=self._outdir/tag):
            return None

        return Deploy(rom_size=self._rom_size(self._outdir/tag/'NucleoH7A3ZIQ.elf', str(self._size_bin)),
                      ram_size=self._ram_size(self._outdir/tag/'NucleoH7A3ZIQ.elf', str(self._size_bin)),
                      evaluator=self.evaluator)
