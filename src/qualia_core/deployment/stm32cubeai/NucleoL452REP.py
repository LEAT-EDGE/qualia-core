from __future__ import annotations

from pathlib import Path

from .STM32CubeAI import STM32CubeAI


class NucleoL452REP(STM32CubeAI):

    def __init__(self, outdir: Path | None = None) -> None:
        outdir = outdir if outdir is not None else Path('out')/'deploy'/'NucleoL452REP'

        super().__init__(outdir=outdir,
                        projectname='STM32CubeAI-NucleoL452REP',
                        projectdir=Path(__file__).parent.parent.parent/'assets'/'projects'/'stm32cubeai'/'NucleoL452REP')
