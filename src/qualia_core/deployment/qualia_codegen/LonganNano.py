from __future__ import annotations

from importlib.resources import files
from typing import TYPE_CHECKING, Any

from qualia_core.deployment.toolchain import NucleiStudio

if TYPE_CHECKING:
    from pathlib import Path


class LonganNano(NucleiStudio):
    import qualia_core.evaluation.target.Qualia as evaluator  # Suggested evaluator

    def __init__(self, projectdir: Path | None = None, *args: Any, **kwargs: Any) -> None:
        projectdir = projectdir if projectdir is not None else files('qualia_codegen_core.examples')/'LonganNano'
        super().__init__(projectname='LonganNano', projectdir=projectdir, *args, **kwargs)

        self.__model_data = self._projectdir/'application'/'full_model.h'

    def __write_model(self, model):
        with self.__model_data.open('w') as f:
            f.write(model.h)

    def prepare(self, tag, model, optimize: str, compression: int):
        if optimize and optimize != 'nmsis-nn':
            raise ValueError(f'Optimization {optimize} not available for {self.__class__.__name__}')
        if compression != 1:
            raise ValueError(f'No compression available for {self.__class__.__name__}')

        self._create_outdir()
        self._clean_workspace()
        self.__write_model(model=model)

        if optimize == 'nmsis-nn':
            args = ('-D', 'WITH_NMSIS_NN',
                    '-D', 'RISCV_MATH_DSP',
                    '-D', 'RISCV_NN_TRUNCATE')
        else:
            args = tuple()

        if not self._build(args=args):
            return None
        self._copy_buildproduct(tag=tag)
        return self
