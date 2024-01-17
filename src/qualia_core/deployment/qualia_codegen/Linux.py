from __future__ import annotations

from collections import namedtuple
from importlib.resources import files
from pathlib import Path

from qualia_core.deployment.Deployer import Deployer
from qualia_core.utils.path import resources_to_path
from qualia_core.utils.process import subprocesstee


class Linux(Deployer):
    import qualia_core.evaluation.host.Qualia as evaluator  # Suggested evaluator

    def __init__(self,
                 cxxflags: list[str] | None = None,
                 modeldir: str | Path | None = None,
                 projectdir: str | Path | None = None,
                 outdir: str | Path | None = None) -> None:
        super().__init__()
        self.__cxxflags = cxxflags if cxxflags is not None else ['-std=c++17',
                                                                 '-Wall', '-Wextra', '-Werror=double-promotion',
                                                                 '-pedantic', '-Ofast', '-ffunction-sections', '-fdata-sections',
                                                                 '-fgraphite-identity', '-floop-nest-optimize',
                                                                 '-floop-parallelize-all',
                                                                 '-fsanitize=signed-integer-overflow',
                                                                 '-fno-sanitize-recover',
                                                                 '-DTRAPV_SHIFT']
        self.__modeldir = Path(modeldir) if modeldir is not None else Path('out')/'qualia_codegen'
        self.__projectdir = (Path(projectdir) if projectdir is not None
                             else resources_to_path(files('qualia_codegen_core.examples'))/'Linux')
        self.__outdir = Path(outdir) if outdir is not None else Path('out')/'deploy'/'Linux'

    def __run(self, cmd, *args):
        print(cmd, *args)
        returncode, outputs = subprocesstee.run(str(cmd), *args)
        return returncode == 0

    def __create_outdir(self):
        self.__outdir.mkdir(parents=True, exist_ok=True)

    def __build(self, tag: str, model):
        modeldir = self.__modeldir/model.name
        return self.__run('g++',
                          '-o', str(self.__outdir/f'{tag}_Linux'),
                           str(self.__projectdir/'main.cpp'),
                           str(modeldir/'model.c'),
                           f'-I{modeldir}',
                           f'-I{modeldir}/include',
                          *self.__cxxflags,
                        )

    def prepare(self, tag, model, optimize: str, compression: int):
        if optimize:
            raise ValueError(f'No optimization available for {self.__class__.__name__}')
        if compression != 1:
            raise ValueError(f'No compression available for {self.__class__.__name__}')

        self.__create_outdir()

        if not self.__build(tag=tag, model=model):
            return None
        return self

    def deploy(self, tag):
        print(self.__class__.__name__, 'Info: running locally, nothing to deploy')

        return namedtuple('Deploy', ['rom_size', 'ram_size', 'evaluator'])(self._rom_size(tag), self._ram_size(tag), self.evaluator)

    def _rom_size(self, tag):
        return super()._rom_size(self.__outdir/f'{tag}_Linux', 'size')

    def _ram_size(self, tag: str):
        return super()._ram_size(self.__outdir/f'{tag}_Linux', 'size')
