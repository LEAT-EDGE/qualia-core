from __future__ import annotations

import logging
import shutil
import sys
from abc import abstractmethod
from typing import Any

from qualia_core.deployment.Deploy import Deploy
from qualia_core.deployment.Deployer import Deployer
from qualia_core.typing import TYPE_CHECKING
from qualia_core.utils.process import subprocesstee

if TYPE_CHECKING:
    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self

    from pathlib import Path  # noqa: TCH003

    from qualia_core.postprocessing.Converter import Converter  # noqa: TCH001

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)

class Eclipse(Deployer):

    def __init__(self,
        eclipse_bin: Path,
        size_bin: Path,
        upload_bin: Path,
        projectname: str,
        projectdir: Path,
        outdir: Path,
        buildtype: str = 'Release') -> None:
        super().__init__()

        self._projectname = projectname
        self._buildtype = buildtype
        self._workspacedir = outdir/'workspace'
        self._projectdir = projectdir

        self._outdir = outdir

        self.__eclipse_bin = eclipse_bin
        self.__size_bin = size_bin
        self.__upload_bin = upload_bin

    def _run(self,
              cmd: str | Path,
              *args: str,
              cwd: Path | None = None,
              env: dict[str, str] | None = None) -> bool:
        logger.info('Running: %s %s', cmd, ' '.join(args))
        returncode, _ = subprocesstee.run(str(cmd), *args, cwd=cwd, env=env)
        return returncode == 0

    def _create_outdir(self) -> None:
        self._outdir.mkdir(parents=True, exist_ok=True)
        self._workspacedir.mkdir(parents=True, exist_ok=True)

    def _clean_workspace(self) -> None:
        shutil.rmtree(self._workspacedir)

    def _build(self, args: tuple[str, ...] | None = None) -> bool:
        return self._run(self.__eclipse_bin,
                            '--launcher.suppressErrors',
                            '-nosplash',
                            '-application', 'org.eclipse.cdt.managedbuilder.core.headlessbuild',
                            '-data', str(self._workspacedir),
                            '-import', str(self._projectdir),
                            '-cleanBuild', f'{self._projectname}/{self._buildtype}',
                            *(args if args is not None else ()),
                        )

    def _copy_buildproduct(self, tag: str) -> None:
        shutil.copy(self._projectdir/self._buildtype/f'{self._projectname}.elf', self._outdir/f'{tag}.elf')

    def _upload(self,
                tag: str,
                logdir: Path,
                args: tuple[str, ...] | None = None,
                cmd: Path | None = None) -> bool:
        cmd = cmd if cmd is not None else self.__upload_bin
        args = args if args is not None else ()
        logger.info('Running: %s %s', cmd, ' '.join(args))
        with (logdir/f'{tag}.txt').open('wb') as logfile:
            _ = logfile.write(' '.join([str(cmd), *args, '\n']).encode('utf-8'))
            returncode, outputs = subprocesstee.run(str(cmd), *args, files={sys.stdout: logfile, sys.stderr: logfile})
        if returncode != 0:
            return False
        if 'Error:' in outputs[1].decode():
            # If output contains the 'Error:' keyword, an error probably happened even though return code may be 0
            return False
        return True

    @abstractmethod
    def prepare(self, tag: str, model: Converter[Any], optimize: str, compression: int) -> Self | None:
        self._create_outdir()
        self._clean_workspace()

        if not self._build():
            return None
        self._copy_buildproduct(tag=tag)
        return self

    @override
    def deploy(self, tag: str) -> Deploy | None:
        logdir = self._outdir/'upload'
        logdir.mkdir(parents=True, exist_ok=True)
        if not self._upload(tag, logdir=logdir):
            return None

        return Deploy(rom_size=self._rom_size(self._outdir/f'{tag}.elf', str(self.__size_bin)),
                      ram_size=self._ram_size(self._outdir/f'{tag}.elf', str(self.__size_bin)),
                      evaluator=self.evaluator)
