from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

from qualia_core.evaluation.Stats import Stats
from qualia_core.typing import TYPE_CHECKING
from qualia_core.utils.process import subprocesstee

from .Evaluator import Evaluator

if TYPE_CHECKING:
    from qualia_core.dataaugmentation.DataAugmentation import DataAugmentation
    from qualia_core.datamodel.RawDataModel import RawDataModel
    from qualia_core.learningframework.LearningFramework import LearningFramework

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)


class STM32CubeAI(Evaluator):
    def __init__(self,
                 stm32cubeai_args: tuple[str, ...] | None = None,
                 mode: str = '') -> None:
        super().__init__()

        self.__mode = mode
        self.__stm32cubeai_args = stm32cubeai_args if stm32cubeai_args is not None else ()

        # Built-in project for 8.1.0
        self.__stm32cubeai_bin = (Path.home() / 'STM32Cube' / 'Repository' / 'Packs' / 'STMicroelectronics'
            / 'X-CUBE-AI' / '8.1.0' / 'Utilities' / 'linux' / 'stm32ai')

    @staticmethod
    def __dataset_to_csv(dataset: RawDataModel, outdir: Path, limit: int | None) -> None:
        test_x = dataset.sets.test.x
        test_y = dataset.sets.test.y
        if limit:
            test_x = test_x[:limit]
            test_y = test_y[:limit]
        test_x = test_x.reshape((test_x.shape[0], -1))
        test_y = test_y.reshape((test_y.shape[0], -1))
        np.savetxt(outdir / 'testX.csv', test_x, delimiter=',', fmt='%f')
        np.savetxt(outdir / 'testY.csv', test_y, delimiter=',', fmt='%f')

    def __validate(self,  # noqa: PLR0913
                   test_x: Path,
                   test_y: Path,
                   modelpath: Path,
                   compression: int | None,
                   outdir: Path,
                   logdir: Path,
                   tag: str) -> tuple[int, dict[int, bytearray]]:

        cmd = str(self.__stm32cubeai_bin)
        args = (
            'validate',
            '--name', 'network',
            '--model', str(modelpath),
            '--mode', self.__mode,
            '--compression', str(compression) if compression else 'none',
            '--valinput', str(test_x),
            '--valoutput', str(test_y),
            '--verbosity', '3',
            '--workspace', str(outdir / 'workspace'),
            '--output', str(outdir / tag),
            '--classifier',
            '--allocate-inputs',
            '--allocate-outputs',
            *self.__stm32cubeai_args,
        )
        logger.info('Running: %s %s', cmd, ' '.join(args))
        with (logdir / f'{tag}.txt').open('wb') as logfile:
            _ = logfile.write(' '.join([str(cmd), *args, '\n']).encode('utf-8'))
            returncode, outputs = subprocesstee.run(cmd, *args, files={sys.stdout: logfile, sys.stderr: logfile})
        return returncode, outputs

    def __parse_validate_stdout(self, s: str) -> tuple[float | None, float | None]:
        duration = re.search(r'[\r,\n]\s*duration\s*:\s*([.\d]+)\s*ms.*$', s, re.MULTILINE)
        if duration is not None:
            duration = float(duration.group(1)) / 1000

        accuracy = re.search(rf'[\r,\n]\s*{self.__mode}\ c-model\ #1\s+([.\d]+)%.*$', s, re.MULTILINE)
        if accuracy is not None:
            accuracy = float(accuracy.group(1)) / 100

        return duration, accuracy

    @override
    def evaluate(self,
                 framework: LearningFramework[Any],
                 model_kind: str,
                 dataset: RawDataModel,
                 target: str,
                 tag: str,
                 limit: int | None = None,
                 dataaugmentations: list[DataAugmentation] | None = None) -> Stats | None:
        if dataaugmentations:
            logger.error('dataaugmentations not supported for %s', self.__class__.__name__)
            raise ValueError

        outdir = Path('out') / 'deploy' / target
        logdir = Path('out') / 'evaluate' / target
        logdir.mkdir(parents=True, exist_ok=True)

        self.__dataset_to_csv(dataset, logdir, limit)

        return_code, outputs = self.__validate(test_x=logdir / 'testX.csv',
                                                test_y=logdir / 'testY.csv',
                                                modelpath=Path('out') / 'deploy' / 'stm32cubeai' / f'{tag}.tflite',
                                                compression=None,
                                                outdir=outdir,
                                                logdir=logdir,
                                                tag=tag)
        if return_code != 0:
            return None
        duration, accuracy = self.__parse_validate_stdout(outputs[1].decode())

        return Stats(avg_time=duration, accuracy=accuracy)
