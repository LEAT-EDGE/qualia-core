from __future__ import annotations

import logging
import re
import shutil
import sys
from pathlib import Path
from typing import Any, cast

from qualia_core.deployment.Deployer import Deployer
from qualia_core.typing import TYPE_CHECKING
from qualia_core.utils.process import subprocesstee

if TYPE_CHECKING:
    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self

    from qualia_core.postprocessing.Converter import Converter  # noqa: TC001

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)

class CMake(Deployer):
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

    def __get_cmake_version(self) -> tuple[int, int, int]:
        _, cmake_version_outputs = subprocesstee.run('cmake', '--version')
        cmake_version_str = re.search(r'([.\d]+)', cmake_version_outputs[1].decode('utf-8'), re.MULTILINE)

        # Decoding into tuple of ints is good enough as CMake versions are specified as <major>.<minor>.<patch>
        if not cmake_version_str:
            logger.warning('Could not find CMake version')
            return (0, 0, 0)

        cmake_version_list = cmake_version_str.group(1).split('.')
        if len(cmake_version_list) != 3:  # noqa: PLR2004
            logger.warning('Could not parse CMake version, expected 3 components, found %d', len(cmake_version_list))
            return (0, 0, 0)

        # Return type is necessarily 3-element tuple as we checked list length just before
        return cast(tuple[int, int, int], tuple(int(d) for d in cmake_version_list))

    def _run_cmake(self, args: tuple[str, ...], projectdir: Path, outdir: Path) -> bool:
        generator = 'Ninja'
        if not shutil.which('ninja'): # Fallback to "make" if "ninja" is not found
            generator = 'Unix Makefiles'

        # --fresh only supported starting from CMake 3.24, otherwise clean build dir manually
        if self.__get_cmake_version() >= (3, 24, 0):
            args = ('--fresh', *args)
        else:
            shutil.rmtree(outdir)

        if not self._run('cmake',
                         '-G', generator,
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
