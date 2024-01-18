from __future__ import annotations

import logging
import sys
from importlib.resources import files
from pathlib import Path
from typing import Any

from qualia_core.deployment.Deploy import Deploy
from qualia_core.deployment.Deployer import Deployer
from qualia_core.evaluation.target.Qualia import Qualia as QualiaEvaluator
from qualia_core.typing import TYPE_CHECKING
from qualia_core.utils.path import resources_to_path
from qualia_core.utils.process import subprocesstee

if TYPE_CHECKING:
    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self

    from qualia_core.postprocessing.Converter import Converter  # noqa: TCH001

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)

class NucleoL452REP(Deployer):
    evaluator = QualiaEvaluator # Suggested evaluator

    def __init__(self) -> None:
        super().__init__()

        self._projectdir = resources_to_path(files('qualia_codegen_core.examples'))/'NucleoL452REP'
        self._outdir = Path('out')/'deploy'/'NucleoL452REP'
        self.__size_bin = 'arm-none-eabi-size'

    def _run(self,
              cmd: str | Path,
              *args: str,
              cwd: Path | None = None,
              env: dict[str, str] | None = None) -> bool:
        logger.info('Running: %s %s', cmd, ' '.join(args))
        returncode, _ = subprocesstee.run(str(cmd), *args, cwd=cwd, env=env)
        return returncode == 0

    def _create_outdir(self, outdir: Path) -> None:
        outdir.mkdir(parents=True, exist_ok=True)

    def _build(self, args: tuple[str, ...], outdir: Path) -> bool:
        if not self._run('cmake',
                         '--fresh',
                         '-G', 'Ninja',
                         '-S', str(self._projectdir.resolve()),
                         '-B', str(outdir.resolve()),
                         *args,
                         cwd=outdir):
            return False
        return self._run('cmake',
                         '--build', str(outdir.resolve()),
                         '--parallel',
                         cwd=outdir)

    @override
    def prepare(self,
                tag: str,
                model: Converter[Any],
                optimize: str,
                compression: int) -> Self | None:
        # Keep here for isinstance() to avoid circual import
        from qualia_core.postprocessing.QualiaCodeGen import QualiaCodeGen

        if optimize and optimize != 'cmsis-nn':
            logger.error('Optimization %s not available for %s', optimize, type(self).__name__)
            raise ValueError

        if compression != 1:
            logger.error('No compression available for %s', type(self).__name__)
            raise ValueError

        if not isinstance(model, QualiaCodeGen):
            logger.error('%s excepts the model to come from a QualiaCodeGen Converter', type(self).__name__)
            raise TypeError

        if model.directory is None:
            logger.error('QualiaCodeGen Converter did not run successfully (QualiaCodeGen.directory is None)')
            raise ValueError

        outdir = self._outdir / tag

        self._create_outdir(outdir)

        args = ('-D', f'MODEL_DIR={model.directory.resolve()!s}')
        if optimize == 'cmsis-nn':
            args = (*args, '-D', 'WITH_CMSIS_NN=True')

        if not self._build(args=args, outdir=outdir):
            return None
        return self

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
