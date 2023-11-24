from abc import abstractmethod
from pathlib import Path
import shutil
import sys
from collections import namedtuple
from qualia_core.utils.process import subprocesstee
from qualia_core.deployment.Deployer import Deployer

class Eclipse(Deployer):

    def __init__(self,
        eclipse_bin: Path,
        size_bin: str,
        upload_bin: Path,
        projectname: str,
        projectdir: Path,
        outdir: Path,
        buildtype: str='Release'):

        self._projectname = projectname
        self._buildtype = buildtype
        self._workspacedir = outdir/'workspace'
        self._projectdir = projectdir

        self._outdir = outdir

        self.__eclipse_bin = eclipse_bin
        self.__size_bin = size_bin
        self.__upload_bin = upload_bin

    def _run(self, cmd, *args):
        print(cmd, *args)
        returncode, outputs = subprocesstee.run(str(cmd), *args)
        return returncode == 0

    def _create_outdir(self):
        self._outdir.mkdir(parents=True, exist_ok=True)
        self._workspacedir.mkdir(parents=True, exist_ok=True)
    
    def _clean_workspace(self):
        shutil.rmtree(self._workspacedir)

    def _build(self, tag: str, args: tuple=()):
        return self._run(self.__eclipse_bin,
                            '--launcher.suppressErrors',
                            '-nosplash',
                            '-application', 'org.eclipse.cdt.managedbuilder.core.headlessbuild',
                            '-data', str(self._workspacedir),
                            '-import', str(self._projectdir),
                            '-cleanBuild', f'{self._projectname}/{self._buildtype}',
                            *args
                        )

    def _copy_buildproduct(self, tag: str):
        shutil.copy(self._projectdir/self._buildtype/f'{self._projectname}.elf', self._outdir/f'{tag}.elf')

    def _upload(self, tag: str, logdir: Path, args: tuple, cmd: Path=None):
        cmd = cmd if cmd is not None else self.__upload_bin
        print(cmd, *args)
        with (logdir/f'{tag}.txt').open('wb') as logfile:
            logfile.write(' '.join([str(cmd), *args, '\n']).encode('utf-8'))
            returncode, outputs = subprocesstee.run(cmd, *args, files={sys.stdout: logfile, sys.stderr: logfile})
        if returncode != 0:
            return False
        if 'Error:' in outputs[1].decode():
            # If output contains the 'Error:' keyword, an error probably happened even though return code may be 0
            return False
        return True

    @abstractmethod
    def prepare(self, tag):
        self._create_outdir()
        self._clean_workspace()

        if not self._build(tag=tag):
            return None
        self._copy_buildproduct(tag=tag)
        return self

    def deploy(self, tag):
        logdir = self._outdir/'upload'
        logdir.mkdir(parents=True, exist_ok=True)
        if not self._upload(tag, logdir=logdir):
            return None

        return namedtuple('Deploy', ['rom_size', 'ram_size', 'evaluator'])(self._rom_size(tag), self._ram_size(tag), self.evaluator)

    def _rom_size(self, tag: str):
        return super()._rom_size(self._outdir/f'{tag}.elf', str(self.__size_bin))

    def _ram_size(self, tag: str):
        return super()._ram_size(self._outdir/f'{tag}.elf', str(self.__size_bin))
