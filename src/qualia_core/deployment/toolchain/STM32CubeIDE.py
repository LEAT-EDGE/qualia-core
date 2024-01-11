from __future__ import annotations

import sys
from pathlib import Path

from .Eclipse import Eclipse

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class STM32CubeIDE(Eclipse):
    def __init__(self,
                 projectname: str,
                 projectdir: Path,
                 outdir: Path | None = None) -> None:
        outdir = outdir if outdir is not None else Path('out')/'deploy'/'STM32CubeIDE'
        stm32cubeide_dir = Path('/opt')/'stm32cubeide'
        stm32cubeide_bin = stm32cubeide_dir/'stm32cubeide'

        # STM32CubeProgrammer bundled with STM32CubeIDE 1.6.0 cannot reset MCU, use system-wide STM32CubeProgrammer 2.7.0
        stm32cubeprogrammer_bin = Path('/')/'usr'/'bin'/'STM32_Programmer_CLI'
        #stm32cubeprogrammer_bin = next((self.__stm32cubeide_dir/'plugins').glob('com.st.stm32cube.ide.mcu.externaltools.cubeprogrammer.linux64_*'))/'tools'/'bin'/'STM32_Programmer.sh'

        arm_size_bin = next(filter(Path.is_dir, (stm32cubeide_dir/'plugins').glob('com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32*')))/'tools'/'bin'/'arm-none-eabi-size'

        super().__init__(eclipse_bin=stm32cubeide_bin,
                         size_bin=arm_size_bin,
                         upload_bin=stm32cubeprogrammer_bin,
                         projectname=projectname,
                         projectdir=projectdir,
                         outdir=outdir)

    @override
    def _upload(self,
                tag: str,
                logdir: Path,
                args: tuple[str, ...] | None = None,
                cmd: Path | None = None) -> bool:
        args = ('--connect', 'port=SWD', 'mode=UR', 'reset=hwRst',
                '--download', str(self._outdir/f'{tag}.elf'), '0x08000000',
                '--verify',
                '-hardRst',
                *(args if args is not None else ()))
        return super()._upload(tag=tag, logdir=logdir, args=args, cmd=cmd)
