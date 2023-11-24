from .Eclipse import Eclipse

from pathlib import Path
import shutil

class NucleiStudio(Eclipse):
    def __init__(self, outdir: Path=Path('out')/'deploy'/'NucleiStudio', *args, **kwargs):
        nuclei_dir = Path('/opt')/'nuclei'
        nucleistudio_bin = nuclei_dir/'NucleiStudio'/'NucleiStudio'

        dfu_util_bin = Path('/usr')/'bin'/'dfu-util'

        riscv_size_bin = nuclei_dir/'gcc'/'bin'/'riscv-nuclei-elf-size'

        super().__init__(eclipse_bin=nucleistudio_bin,
                       size_bin=riscv_size_bin,
                       upload_bin=dfu_util_bin,
                       outdir=outdir,
                       *args, **kwargs)

    def _copy_buildproduct(self, tag: str):
        shutil.copy(self._projectdir/self._buildtype/f'{self._projectname}.bin', self._outdir/f'{tag}.bin')
        return super()._copy_buildproduct(tag=tag)

    def _upload(self, tag: str, logdir: Path):
        args = ('-s', '0x08000000:leave', '-D', str(self._outdir/f'{tag}.bin'))
        return super()._upload(tag=tag, logdir=logdir, args=args)

    def deploy(self, *args, **kwargs):
        input('Put target in programming mode and press Enter…')
        ret = super().deploy(*args, **kwargs)
        return ret
