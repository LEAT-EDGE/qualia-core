from __future__ import annotations

from collections import namedtuple
from importlib.resources import files
from pathlib import Path

from qualia_core.deployment.Deployer import Deployer
from qualia_core.utils.process import subprocesstee


class SparkFunEdge(Deployer):
    import qualia_core.evaluation.target.Qualia as evaluator  # Suggested evaluator

    def __init__(self,
                 dev: str = '/dev/ttyUSB0',
                 modeldir: Path = Path('out')/'qualia_codegen',
                 projectdir: Path | None = None,
                 outdir: Path = Path('out')/'deploy'/'SparkFunEdge') -> None:
        super().__init__()
        self.__dev = dev
        self.__modeldir = modeldir
        self.__projectdir = projectdir if projectdir is not None else files('qualia_codegen_core.examples')/'SparkFunEdge'
        self.__outdir = outdir

    def __run(self, cmd, *args):
        print(cmd, *args)
        returncode, outputs = subprocesstee.run(str(cmd), *args)
        return returncode == 0

    def __create_outdir(self):
        self.__outdir.mkdir(parents=True, exist_ok=True)

    def __clean(self, tag: str, model):
        modeldir = self.__modeldir/model.name
        outdir = self.__outdir/tag
        return self.__run('make',
                          '-C', str(self.__projectdir),
                          f'MODELDIR={modeldir.absolute()}',
                          f'OUT={outdir.absolute()}',
                          'clean'
                        )

    def __build(self, tag: str, model, optimize: str):
        modeldir = self.__modeldir/model.name
        outdir = self.__outdir/tag
        args = ['-C', str(self.__projectdir),
                f'MODELDIR={modeldir.absolute()}',
                f'OUT={outdir.absolute()}']
        if optimize == 'cmsis-nn':
            args.append('WITH_CMSIS_NN=1')
        return self.__run('make', *args)

    def __upload(self, tag: str):
        outdir = self.__outdir/tag
        return self.__run('make',
                          '-C', str(self.__projectdir),
                          f'OUT={outdir.absolute()}',
                          f'SERIAL_PORT={self.__dev}',
                          'bootload'
                        )

    def prepare(self, tag, model, optimize: str, compression: int):
        if optimize and optimize != 'cmsis-nn':
            raise ValueError(f'Optimization {optimize} not available for {self.__class__.__name__}')
        if compression != 1:
            raise ValueError(f'No compression available for {self.__class__.__name__}')

        print('model:', model)

        self.__create_outdir()

        self.__clean(tag=tag, model=model)
        if not self.__build(tag=tag, model=model, optimize=optimize):
            return None
        return self

    def deploy(self, tag):
        input('Put target in programming mode and press Enter…')
        self.__upload(tag=tag)

        return namedtuple('Deploy', ['rom_size', 'ram_size', 'evaluator'])(self._rom_size(tag), self._ram_size(tag), self.evaluator)

    def _rom_size(self, tag):
        return super()._rom_size(self.__outdir/tag/'main.axf', 'arm-none-eabi-size')

    def _ram_size(self, tag):
        return super()._ram_size(self.__outdir/tag/'main.axf', 'arm-none-eabi-size')
