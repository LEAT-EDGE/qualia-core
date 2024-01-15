from __future__ import annotations

import logging
import sys
from importlib.resources import files
from typing import Any

from qualia_core.deployment.toolchain import NucleiStudio
from qualia_core.evaluation.target.Qualia import Qualia as QualiaEvaluator
from qualia_core.typing import TYPE_CHECKING
from qualia_core.utils.path import resources_to_path

if TYPE_CHECKING:
    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self

    from qualia_core.postprocessing.Converter import Converter  # noqa: TCH001
    from qualia_core.postprocessing.QualiaCodeGen import QualiaCodeGen  # noqa: TCH001

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)

class LonganNano(NucleiStudio):
    evaluator = QualiaEvaluator # Suggested evaluator

    def __init__(self) -> None:
        projectdir = resources_to_path(files('qualia_codegen_core.examples'))/'LonganNano'
        super().__init__(projectname='LonganNano',
                         projectdir=projectdir)

        self.__model_data = self._projectdir/'application'/'full_model.h'

    def __write_model(self, model: QualiaCodeGen) -> None:
        if model.h is None:
            logger.error('Cannot write model source: QualiaCodeGen Converter did not run successfully (QualiaCodeGen.h is None)')
            raise ValueError

        with self.__model_data.open('w') as f:
            _ = f.write(model.h)

    @override
    def prepare(self,
                tag: str,
                model: Converter[Any],
                optimize: str,
                compression: int) -> Self | None:
        # Keep here for isinstance() to avoid circual import
        from qualia_core.postprocessing.QualiaCodeGen import QualiaCodeGen

        if optimize and optimize != 'nmsis-nn':
            logger.error('Optimization %s not available for %s', optimize, type(self).__name__)
            raise ValueError

        if compression != 1:
            logger.error('No compression available for %s', type(self).__name__)
            raise ValueError

        if not isinstance(model, QualiaCodeGen):
            logger.error('%s excepts the model to come from a QualiaCodeGen Converter', type(self).__name__)
            raise TypeError

        if model.directory is None:
            logger.error('QualiaCodeGen Converter did not run successfully (QualiaCodeGen.directory is None)')
            raise ValueError

        self._create_outdir()
        self._clean_workspace()
        self.__write_model(model=model)

        args: tuple[str, ...] = ('-include', str(model.directory.resolve()/'include'/'defines.h'))

        if optimize == 'nmsis-nn':
            args = (*args,
                    '-D', 'WITH_NMSIS_NN')

        if not self._build(args=args):
            return None
        self._copy_buildproduct(tag=tag)
        return self
