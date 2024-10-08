from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from qualia_core.deployment.Deployer import Deployer
from qualia_core.evaluation.target.Qualia import Qualia as QualiaEvaluator
from qualia_core.typing import TYPE_CHECKING
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

class CMake(Deployer):
    evaluator = QualiaEvaluator # Suggested evaluator

    def __init__(self,
                 projectdir: str | Path,
                 outdir: str | Path) -> None:
        super().__init__()

        self._projectdir = Path(projectdir)
        self._outdir = Path(outdir)

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

    def _run_cmake(self, args: tuple[str, ...], projectdir: Path, outdir: Path) -> bool:
        if not self._run('cmake',
                         '--fresh',
                         '-G', 'Ninja',
                         '-S', str(projectdir.resolve()),
                         '-B', str(outdir.resolve()),
                         *args,
                         cwd=outdir):
            return False
        return self._run('cmake',
                         '--build', str(outdir.resolve()),
                         '--parallel',
                         cwd=outdir)

    def _build(self,
               modeldir: Path,
               optimize: str,
               outdir: Path) -> bool:
        args = ('-D', f'MODEL_DIR={modeldir.resolve()!s}')

        return self._run_cmake(args=args, projectdir=self._projectdir, outdir=outdir)

    def _validate_optimize(self, optimize: str) -> None:
        if optimize:
            logger.error('No optimization available for %s', type(self).__name__)
            raise ValueError
    def _validate_compression(self, compression: int) -> None:
        if compression != 1:
            logger.error('No compression available for %s', type(self).__name__)
            raise ValueError

    @override
    def prepare(self,
                tag: str,
                model: Converter[Any],
                optimize: str,
                compression: int) -> Self | None:
        # Keep here for isinstance() to avoid circual import
        from qualia_core.postprocessing.QualiaCodeGen import QualiaCodeGen

        if not isinstance(model, QualiaCodeGen):
            logger.error('%s excepts the model to come from a QualiaCodeGen Converter', type(self).__name__)
            raise TypeError

        if model.directory is None:
            logger.error('QualiaCodeGen Converter did not run successfully (QualiaCodeGen.directory is None)')
            raise ValueError

        self._validate_optimize(optimize)
        self._validate_compression(compression)

        outdir = self._outdir / tag

        self._create_outdir(outdir)

        if not self._build(modeldir=model.directory, optimize=optimize, outdir=outdir):
            return None

        return self
