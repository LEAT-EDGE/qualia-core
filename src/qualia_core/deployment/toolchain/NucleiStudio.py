from __future__ import annotations

import shutil
import sys
from pathlib import Path

from qualia_core.typing import TYPE_CHECKING

from .Eclipse import Eclipse

if TYPE_CHECKING:
    from qualia_core.deployment.Deploy import Deploy  # noqa: TCH001

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class NucleiStudio(Eclipse):
    def __init__(self,
                 projectname: str,
                 projectdir: Path,
                 outdir: Path | None = None) -> None:
        outdir = outdir if outdir is not None else Path('out')/'deploy'/'NucleiStudio'

        nuclei_dir = Path('/opt')/'nuclei'
        nucleistudio_bin = nuclei_dir/'NucleiStudio'/'NucleiStudio'

        dfu_util_bin = Path('/usr')/'bin'/'dfu-util'

        riscv_size_bin = nuclei_dir/'gcc'/'bin'/'riscv-nuclei-elf-size'

        super().__init__(eclipse_bin=nucleistudio_bin,
                         size_bin=riscv_size_bin,
                         upload_bin=dfu_util_bin,
                         projectname=projectname,
                         projectdir=projectdir,
                         outdir=outdir)

    @override
    def _copy_buildproduct(self, tag: str) -> None:
        shutil.copy(self._projectdir/self._buildtype/f'{self._projectname}.bin', self._outdir/f'{tag}.bin')
        return super()._copy_buildproduct(tag=tag)

    @override
    def _upload(self,
                tag: str,
                logdir: Path,
                args: tuple[str, ...] | None = None,
                cmd: Path | None = None) -> bool:
        args = ('-s', '0x08000000:leave', '-D', str(self._outdir/f'{tag}.bin'))
        return super()._upload(tag=tag, logdir=logdir, args=args)

    @override
    def deploy(self, tag: str) -> Deploy | None:
        _ = input('Put target in programming mode and press Enter…')
        return super().deploy(tag=tag)
