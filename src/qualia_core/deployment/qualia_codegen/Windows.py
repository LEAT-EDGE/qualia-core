from __future__ import annotations

import logging
import sys

from qualia_core.deployment.Deploy import Deploy
from qualia_core.evaluation.host.Qualia import Qualia as QualiaEvaluator

from .Linux import Linux

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)

class Windows(Linux):
    evaluator = QualiaEvaluator # Suggested evaluator

    @override
    def deploy(self, tag: str) -> Deploy | None:
        logger.info('Running locally, nothing to deploy')

        return Deploy(rom_size=self._rom_size(self._outdir/tag/'Linux.exe', str(self._size_bin)),
                      ram_size=self._ram_size(self._outdir/tag/'Linux.exe', str(self._size_bin)),
                      evaluator=self.evaluator)
