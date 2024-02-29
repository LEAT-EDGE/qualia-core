from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple

import colorful as cf  # type: ignore[import-untyped]

from qualia_core.qualia import train
from qualia_core.typing import TYPE_CHECKING
from qualia_core.utils.logger import CSVLogger, Logger

if TYPE_CHECKING:
    from qualia_core.dataaugmentation.DataAugmentation import DataAugmentation  # noqa: TCH001
    from qualia_core.datamodel.RawDataModel import RawDataModel  # noqa: TCH001
    from qualia_core.learningframework.LearningFramework import LearningFramework  # noqa: TCH001
    from qualia_core.typing import ConfigDict
    from qualia_core.utils import Git  # noqa: TCH001
    from qualia_core.utils.plugin import QualiaComponent  # noqa: TCH001

logger = logging.getLogger(__name__)

class LearningModelLoggerFields(NamedTuple):
    name: str
    i: int
    params: int
    mem_params: int
    accuracy: float

class Train:
    def __call__(self,  # noqa: PLR0913
                 qualia: QualiaComponent,
                 learningframework: LearningFramework[Any],
                 dataaugmentations: list[DataAugmentation],
                 data: RawDataModel,
                 config: ConfigDict,
                 git: Git) -> dict[str, Logger[LearningModelLoggerFields]]:
        loggers: dict[str, Logger[LearningModelLoggerFields]] = {}

        log: CSVLogger[LearningModelLoggerFields] = CSVLogger('learningmodel',
                                                              file=Path('learningmodel')
                                                                   /f'{git.short_hash()}_{datetime.now():%Y-%m-%d_%H-%M-%S}.csv')
        loggers['learningmodel'] = log
        log.fields = LearningModelLoggerFields # Write column names

        experimenttracking = config.get('experimenttracking', None)

        (Path('out')/'learningmodel').mkdir(parents=True, exist_ok=True)

        for i in range(config['bench']['first_run'], config['bench']['last_run']+1):
            for m in config['model']:
                if m.get('disabled', False):
                    continue

                if m.get('load', False):
                    print(f'{cf.bold}Loading {cf.blue}{m["name"]}{cf.close_fg_color}, run {cf.red}{i}{cf.reset}')
                else:
                    print(f'{cf.bold}Training {cf.blue}{m["name"]}{cf.close_fg_color}, run {cf.red}{i}{cf.reset}')
                print(f'{cf.bold}Params:{cf.reset} {m=}')

                et = None
                if experimenttracking is not None:
                    et = getattr(learningframework.experimenttrackings, experimenttracking['kind'])(**experimenttracking.get('params', {}))
                    et.start()
                    et.hyperparameters = {'config': config, 'model': m, 'i': i}

                model = getattr(learningframework.learningmodels, m['kind'], None)
                if model is None:
                    logger.error("Could not load model.kind '%s' from learningmodels '%s'.",
                                 m['kind'],
                                 learningframework.learningmodels.__name__)
                    logger.error("Did you load the necessary plugins (loaded: %s) and use the correct learningframework.kind (in use: '%s')?",
                                 config['bench'].get('plugins', []),
                                 learningframework.__class__.__name__)
                    raise ModuleNotFoundError


                trainresult = train(datamodel=data,
                        train_epochs=m.get('epochs', 0),
                        iteration=i,
                        model_name=m['name'],
                        model=model,
                        model_params=m.get('params', {}),
                        batch_size=m.get('batch_size', None),
                        optimizer=m.get('optimizer', None),
                        framework=learningframework,
                        load=m.get('load', False),
                        train=m.get('train', True),
                        evaluate=m.get('evaluate', True),
                        dataaugmentations=dataaugmentations,
                        experimenttracking=et,
                        use_test_as_valid=config['bench'].get('use_test_as_valid', False))

                if et is not None:
                    et.stop()

                log(LearningModelLoggerFields(name=trainresult.name,
                                              i=i,
                                              params=trainresult.params,
                                              mem_params=trainresult.mem_params,
                                              accuracy=trainresult.acc))

                for postprocessing in config.get('postprocessing', []):
                    ppp = {k: v for k,v in postprocessing.get('params', {}).items()} # Workaround tomlkit bug where some nested dict would lose their items
                    trainresult, m = getattr(qualia.postprocessing, postprocessing['kind'])(**ppp)(
                        trainresult=trainresult,
                        model_conf=m)
                    if trainresult.log:
                        log(LearningModelLoggerFields(name=trainresult.name,
                                                      i=i,
                                                      params=trainresult.params,
                                                      mem_params=trainresult.mem_params,
                                                      accuracy=trainresult.acc))
                    if postprocessing.get('export', False):
                        trainresult.framework.export(trainresult.model, f'{trainresult.name}_r{i}')

        return loggers
