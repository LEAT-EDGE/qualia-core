from __future__ import annotations

import logging
import sys
from importlib.resources import files
from pathlib import Path
from threading import Thread

from qualia_core.deployment.Deploy import Deploy
from qualia_core.evaluation.target.Qualia import Qualia as QualiaEvaluator
from qualia_core.utils.path import resources_to_path

from .CMake import CMake

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)

class NucleoH7S3L8(CMake):
    evaluator = QualiaEvaluator # Suggested evaluator

    def __init__(self,
                 projectdir: str | Path | None = None,
                 outdir: str | Path | None = None,
                 extflash: bool = False,  # noqa: FBT001, FBT002
                 sram: bool = True,  # noqa: FBT001, FBT002
                 core_clock_740mhz: bool = False) -> None:
        if projectdir is None:
            if extflash:
                projectdir = resources_to_path(files('qualia_codegen_core.examples'))/'NucleoH7S3L8ExtFlash'
            else:
                projectdir = resources_to_path(files('qualia_codegen_core.examples'))/'NucleoH7S3L8'

        super().__init__(projectdir=projectdir,
                         outdir=outdir if outdir is not None else Path('out')/'deploy'/'NucleoH7S3L8')

        self.__size_bin = 'arm-none-eabi-size'
        self.__extflash = extflash
        self.__sram = sram
        self.__core_clock_740mhz = core_clock_740mhz

    @override
    def _validate_optimize(self, optimize: str) -> None:
        if optimize and optimize != 'cmsis-nn':
            logger.error('Optimization %s not available for %s', optimize, type(self).__name__)
            raise ValueError

    @override
    def _build(self,
               modeldir: Path,
               optimize: str,
               outdir: Path) -> bool:
        args = ('-D', f'MODEL_DIR={modeldir.resolve()!s}')
        if optimize == 'cmsis-nn':
            args = (*args, '-D', 'WITH_CMSIS_NN=True')
        if self.__sram:
            args = (*args, '-D', 'ROM_IN_SRAM=True')
        if self.__core_clock_740mhz:
            args = (*args, '-D', 'CORE_CLOCK_740MHZ=True')

        return self._run_cmake(args=args, projectdir=self._projectdir, outdir=outdir)

    def __run_openocd(self) -> bool:
        return self._run('openocd',
                         '-f', 'interface/stlink-dap.cfg',
                         '-f', 'target/stm32h7rx.cfg')

    def __run_gdb(self, elf: Path) -> bool:
        return self._run('arm-none-eabi-gdb',
                         str(elf),
                         '-ex', 'set confirm off',
                         '-ex', 'target remote localhost:3333',
                         '-ex', 'monitor [target current] configure -event gdb-detach {shutdown}',
                         '-ex', 'monitor reset halt',
                         '-ex', 'load',
                         '-ex', 'continue&',
                         '-ex', 'quit')

    @override
    def deploy(self, tag: str) -> Deploy | None:
        # if not self._run('openocd',
        #                  '-f', 'interface/stlink.cfg',
        #                  '-f', 'target/stm32h7x.cfg',
        #                  '-c', 'init',
        #                  '-c', 'reset halt; flash write_image erase ./NucleoH7S3L8; reset; shutdown',
        #                  cwd=self._outdir/tag):

        # Flash Boot to MCU's internal Flash
        elf = self._outdir/tag/'NucleoH7S3L8' if not self.__extflash else self._outdir/tag/'NucleoH7S3L8ExtFlash_Boot'
        elf = elf.rename(elf.with_suffix('.elf')) if elf.exists() else elf.with_suffix('.elf')

        if self.__sram:
            openocd_thread = Thread(target=self.__run_openocd)
            openocd_thread.start()
            gdb_thread = Thread(target=self.__run_gdb, kwargs={'elf': elf})
            gdb_thread.start()
            #gdb_thread.join()
            #openocd_thread.join()

        else:
            if not self._run('STM32_Programmer_CLI',
                             '--connect', 'port=SWD', 'mode=UR', 'reset=hwRst',
                             '--download', str(elf),# '0x08000000',
                             '--verify',
                             '-hardRst'):
                return None

            # Flash Appli to external Flash
            if self.__extflash:
                elf = self._outdir/tag/'NucleoH7S3L8ExtFlash_Appli'
                elf = elf.rename(elf.with_suffix('.elf')) if elf.exists() else elf.with_suffix('.elf')

                if not self._run('STM32_Programmer_CLI',
                                 '--connect', 'port=SWD', 'mode=UR', 'reset=hwRst',
                                 '--extload', '/opt/stm32cubeprog/bin/ExternalLoader/MX25UW25645G_NUCLEO-H7S3L8.stldr',
                                 '--download', str(elf),# '0x70000000',
                                 '--verify',
                                 '-hardRst'):
                    return None

        return Deploy(rom_size=self._rom_size(elf, str(self.__size_bin)),
                      ram_size=self._ram_size(elf, str(self.__size_bin)),
                      evaluator=self.evaluator)
