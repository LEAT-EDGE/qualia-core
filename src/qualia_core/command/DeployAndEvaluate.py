from __future__ import annotations

import itertools
from datetime import datetime
from pathlib import Path
from typing import Any

import colorful as cf  # type: ignore[import-untyped]

from qualia_core.evaluation.Stats import StatsFields
from qualia_core.qualia import deploy, evaluate, gen_tag, instantiate_model
from qualia_core.typing import TYPE_CHECKING
from qualia_core.utils.logger import CSVLogger, Logger

if TYPE_CHECKING:
    from types import ModuleType  # noqa: TCH003

    from qualia_core.dataaugmentation.DataAugmentation import DataAugmentation  # noqa: TCH001
    from qualia_core.datamodel.RawDataModel import RawDataModel  # noqa: TCH001
    from qualia_core.learningframework.LearningFramework import LearningFramework  # noqa: TCH001
    from qualia_core.postprocessing.Converter import Converter  # noqa: TCH001
    from qualia_core.typing import ConfigDict
    from qualia_core.utils import Git  # noqa: TCH001
    from qualia_core.utils.plugin import QualiaComponent  # noqa: TCH001


class DeployAndEvaluate:
    def __call__(self,  # noqa: PLR0913
                 qualia: QualiaComponent,
                 learningframework: LearningFramework[Any],
                 dataaugmentations: list[DataAugmentation],
                 converter: type[Converter],
                 deployers: ModuleType,
                 data: RawDataModel,
                 config: ConfigDict,
                 git: Git) -> dict[str, Logger[StatsFields]]:
        loggers: dict[str, Logger[StatsFields]] = {}
        log: CSVLogger[StatsFields] = CSVLogger('evaluate', file=Path('evaluate')/f'{git.short_hash()}_{datetime.now():%Y-%m-%d_%H-%M-%S}.csv')
        loggers['evaluate'] = log
        # Write column names
        log.fields = StatsFields

        for i in range(config['bench']['first_run'], config['bench']['last_run']+1):
            for m, q, o, c in itertools.product(config['model'],
                                                config['deploy']['quantize'],
                                                config['deploy']['optimize'],
                                                config['deploy'].get('compress', [1])):
                if m.get('disabled', False):
                    continue

                # Postprocessings can change model name, frameworks, mem_params
                model_name = m['name']
                fmem_params = lambda framework, model: framework.n_params(model) * 4 # By default models have 4-bytes params
                for postprocessing in config.get('postprocessing', []):
                    ppp = {k: v for k,v in postprocessing.get('params', {}).items()} # Workaround tomlkit bug where some nested dict would lose their items
                    pp = getattr(qualia.postprocessing, postprocessing['kind'])(**ppp)
                    model_name = pp.process_name(model_name)
                    learningframework = pp.process_framework(learningframework)
                    #mem_params = pp.process_mem_params(mem_params) #FIXME: handled by converter instead

                #model = framework.load(f'{model_name}_r{i}', m['kind'])


                # Instantiate model
                model = instantiate_model(dataset=data.sets.test,
                                          framework=learningframework,
                                          model=getattr(learningframework.learningmodels, m['kind']),
                                          model_params=m.get('params', {}),
                                          model_name=model_name,
                                          iteration=i,
                                          load=False # Model params will be loaded after postprocessings
                                          )

                # Postprocessings can change model topology with PyTorch, needs to be done after instantiating model with new name
                for postprocessing in config.get('postprocessing', []):
                    pp = getattr(qualia.postprocessing, postprocessing['kind'])(**postprocessing.get('params', {}))
                    model, m = pp.process_model(model, m)

                # Show model architecture
                learningframework.summary(model)

                # Load weights after topology optionally changed
                model = learningframework.load(f'{model_name}_r{i}', model)

                # Converter can affect mem_params if there is quantization
                if not converter:
                    converter = getattr(m['kind'], 'converter', None)
                fmem_params = converter(quantize=q).process_mem_params(fmem_params) if converter else fmem_params

                tag = gen_tag(model_name, q, o, i, c)
                print(f'{cf.bold}Deploying {cf.blue}{model_name}{cf.close_fg_color}, run {cf.red}{i}{cf.close_fg_color}, tag {cf.yellow}{tag}{cf.reset}')
                result_deploy = deploy(model_kind=m['kind'],
                                    deploy_target=config['deploy']['target'],
                                    tag=tag,
                                    deployers=deployers,
                                    deployer_params=config['deploy'].get('deployer', {}).get('params', {}))
                if not result_deploy:
                    continue

                print(f'{cf.bold}Evaluating {cf.blue}{model_name}{cf.close_fg_color}, run {cf.red}{i}{cf.close_fg_color}, tag {cf.yellow}{tag}{cf.reset}')
                result = evaluate(datamodel=data,
                                model_kind=m['kind'],
                                model_name=model_name,
                                model=model,
                                framework=learningframework,
                                iteration=i,
                                target=config['deploy']['target'],
                                quantization=q,
                                fmem_params=fmem_params,
                                tag=tag,
                                limit=config['deploy'].get('limit', None),
                                evaluator=getattr(result_deploy,
                                                   'evaluator',
                                                   getattr(converter, 'evaluator', None) if converter else None),
                                evaluator_params=config['deploy'].get('evaluator', {}).get('params', {}),
                                dataaugmentations=dataaugmentations)

                # fill in ROM and RAM size from deployment result
                if result:
                    result.rom_size = result_deploy.rom_size
                    result.ram_size = result_deploy.ram_size

                print(result)
                if result is not None:
                    log(result.asnamedtuple())
        return loggers
