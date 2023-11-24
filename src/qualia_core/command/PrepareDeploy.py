from __future__ import annotations

import itertools
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple

import colorful as cf  # type: ignore[import-untyped]

from qualia_core.qualia import gen_tag, instantiate_model, prepare_deploy
from qualia_core.typing import TYPE_CHECKING
from qualia_core.utils.logger import CSVLogger, Logger

if TYPE_CHECKING:
    from types import ModuleType  # noqa: TCH003

    from qualia_core.datamodel.RawDataModel import RawDataModel  # noqa: TCH001
    from qualia_core.learningframework.LearningFramework import LearningFramework  # noqa: TCH001
    from qualia_core.postprocessing.Converter import Converter  # noqa: TCH001
    from qualia_core.typing import ConfigDict
    from qualia_core.utils import Git  # noqa: TCH001
    from qualia_core.utils.plugin import QualiaComponent  # noqa: TCH001

class PrepareDeployLoggerFields(NamedTuple):
    name: str
    quantize: str
    optimize: str
    compress: int

class PrepareDeploy:
    def __call__(self,  # noqa: PLR0913
                 qualia: QualiaComponent,
                 learningframework: LearningFramework[Any],
                 converter: type[Converter],
                 deployers: ModuleType,
                 data: RawDataModel,
                 config: ConfigDict,
                 git: Git) -> dict[str, Logger[PrepareDeployLoggerFields]]:
        loggers: dict[str, Logger[PrepareDeployLoggerFields]] = {}

        log: CSVLogger[PrepareDeployLoggerFields] = CSVLogger('prepare_deploy',
                                                             file=Path('prepare_deploy')/f'{git.short_hash()}_{datetime.now():%Y-%m-%d_%H-%M-%S}.csv')
        loggers['prepare_deploy'] = log
        # Write column names
        log.fields = PrepareDeployLoggerFields

        for i in range(config['bench']['first_run'], config['bench']['last_run']+1):
            for m, q, o, c in itertools.product(config['model'],
                                                config['deploy']['quantize'],
                                                config['deploy']['optimize'],
                                                config['deploy'].get('compress', [1])):
                if m.get('disabled', False):
                    continue

                # Postprocessings can change model name
                model_name = m['name']
                for postprocessing in config.get('postprocessing', []):
                    ppp = {k: v for k,v in postprocessing.get('params', {}).items()} # Workaround tomlkit bug where some nested dict would lose their items
                    pp = getattr(qualia.postprocessing, postprocessing['kind'])(**ppp)
                    model_name = pp.process_name(model_name)
                    learningframework = pp.process_framework(learningframework)

                # Instantiate model
                model = instantiate_model(dataset=data.sets.test,
                                          framework=learningframework,
                                          model=getattr(learningframework.learningmodels, m['kind']),
                                          model_params=m.get('params', {}),
                                          model_name=model_name,
                                          iteration=i,
                                          load=False, # Model params will be loaded after postprocessings
                                          )

                # Postprocessings can change model topology with PyTorch, needs to be done after instantiating model with new name
                for postprocessing in config.get('postprocessing', []):
                    pp = getattr(qualia.postprocessing, postprocessing['kind'])(**postprocessing.get('params', {}))
                    model, m = pp.process_model(model, m)

                # Show model architecture
                learningframework.summary(model)

                # Load weights after topology optionally changed
                model = learningframework.load(f'{model_name}_r{i}', model)

                print(f'{cf.bold}Preparing {cf.blue}{model_name}{cf.close_fg_color}, run {cf.red}{i}{cf.reset}')
                r = prepare_deploy(
                               datamodel=data,
                               model_kind=m['kind'],
                               model_name=model_name,
                               model=model,
                               framework=learningframework,
                               iteration=i,
                               quantize=q,
                               optimize=o,
                               compress=c,
                               deploy_target=config['deploy']['target'],
                               tag=gen_tag(model_name, q, o, i, c),
                               converter=converter,
                               converter_params=config['deploy'].get('converter', {}).get('params', {}),
                               deployers=deployers,
                               deployer_params=config['deploy'].get('deployer', {}).get('params', {}),
                               representative_dataset=data.sets.train.x)
                if not r:
                    continue

                log(PrepareDeployLoggerFields(name=model_name, quantize=q, optimize=o, compress=c))

        return loggers
