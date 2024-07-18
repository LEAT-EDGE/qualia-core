from __future__ import annotations

import logging
import sys
from importlib.resources import files
from pathlib import Path

from qualia_core.deployment.Deploy import Deploy
from qualia_core.evaluation.host.Qualia import Qualia as QualiaEvaluator
from qualia_core.utils.path import resources_to_path

from .CMake import CMake

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)

class Linux(CMake):
    evaluator = QualiaEvaluator # Suggested evaluator

    def __init__(self) -> None:
        super().__init__(projectdir=resources_to_path(files('qualia_codegen_core.examples'))/'Linux',
                         outdir=Path('out')/'deploy'/'Linux')

        self.__size_bin = 'size'

    @override
    def deploy(self, tag: str) -> Deploy | None:
        logger.info('Running locally, nothing to deploy')

        return Deploy(rom_size=self._rom_size(self._outdir/tag/'Linux', str(self.__size_bin)),
                      ram_size=self._ram_size(self._outdir/tag/'Linux', str(self.__size_bin)),
                      evaluator=self.evaluator)
