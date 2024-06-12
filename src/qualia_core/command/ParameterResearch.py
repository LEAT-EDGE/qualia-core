from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple

import colorful as cf  # type: ignore[import-untyped]

from qualia_core.typing import TYPE_CHECKING
from qualia_core.utils.merge_dict import merge_dict

if TYPE_CHECKING:
    from qualia_core.dataaugmentation.DataAugmentation import DataAugmentation  # noqa: TCH001
    from qualia_core.datamodel.RawDataModel import RawDataModel  # noqa: TCH001
    from qualia_core.learningframework.LearningFramework import LearningFramework  # noqa: TCH001
    from qualia_core.typing import ConfigDict
    from qualia_core.utils import Git  # noqa: TCH001
    from qualia_core.utils.logger import Logger  # noqa: TCH001
    from qualia_core.utils.plugin import QualiaComponent  # noqa: TCH001

class ParameterResearchLoggerFields(NamedTuple):
    name: str
    params: int
    mem_params: int
    accuracy: float
    tune_config: str

class ParameterResearch:
    def __call__(self,
                 qualia: QualiaComponent,
                 learningframework: LearningFramework[Any],
                 dataaugmentations: list[DataAugmentation],
                 data: RawDataModel,
                 config: ConfigDict,
                 git: Git) -> dict[str, Logger[ParameterResearchLoggerFields]]:
        import optuna

        from qualia_core.qualia import train
        from qualia_core.utils.logger import CSVLogger

        loggers: dict[str, CSVLogger[ParameterResearchLoggerFields]] = {}

        log: CSVLogger[ParameterResearchLoggerFields] = CSVLogger('parameter_research',
                                                                  file=Path('parameter_research')/f'{git.short_hash()}_{datetime.now():%Y-%m-%d_%H-%M-%S}.csv')
        loggers['parameter_research'] = log
        # Write column names
        log.fields = ParameterResearchLoggerFields

        (Path('out')/'learningmodel').mkdir(parents=True, exist_ok=True)

        experimenttracking = config.get('experimenttracking', None)

        if config['bench']['first_run'] != config['bench']['last_run']:
            print('Warning: parameter_research does not follow first_run/last_run')

        # Filter out disabled models
        config['model'] = [m for m in config['model'] if not m.get('disabled', False)]

        if isinstance(config['model'], list):
            if len(config['model']) > 1:
                raise ValueError('parameter_research only runs on a single model definition')
            else:
                config['model'] = config['model'][0]
        if config['model'].get('disabled', False):
            raise ValueError('model must not be disabled for parameter_research')


        def training_function(trial):
            def gen_tune_config(params): # In-place
                if isinstance(params, list):
                    for i, v in enumerate(params):
                        if 'kind' in v and 'params' in v: # "leaf" to call trial function on
                            params[i] = getattr(trial, v['kind'])(**v['params'])
                        else:
                            gen_tune_config(i)
                else:
                    print(params)
                    for k, v in params.items():
                        if 'kind' in v and 'params' in v: # "leaf" to call trial function on
                            params[k] = getattr(trial, v['kind'])(**v['params'])
                        else:
                            gen_tune_config(v)
                return params

            print(f'{cf.bold}Hyperparameter research for {cf.blue}{config["model"]["name"]}{cf.close_fg_color}, trial {cf.red}{trial.number}{cf.reset}')

            tune_config = gen_tune_config(copy.deepcopy(config['parameter_research']['trial']))

            print(f'{cf.bold}Trial params:{cf.reset} {tune_config=}')

            m = copy.deepcopy(tune_config)
            m = merge_dict(m, config['model'])

            print(f'{cf.bold}Model params:{cf.reset} {m=}')

            et = None
            if experimenttracking is not None:
                et = getattr(learningframework.experimenttrackings, experimenttracking['kind'])(**experimenttracking.get('params', {}))
                et.hyperparameters = {'config': config, 'model': m, 'i': trial.number}


            trainresult = train(datamodel=data,
                    train_epochs=m.get('epochs', None),
                    iteration=trial.number,
                    model_name=m["name"],
                    model=getattr(learningframework.learningmodels, m['kind']),
                    model_params=m.get('params', {}),
                    batch_size=m.get('batch_size', None),
                    optimizer=m.get('optimizer', {}),
                    framework=learningframework,
                    load=m.get('load', False),
                    train=m.get('train', True),
                    evaluate=m.get('evaluate', True),
                    dataaugmentations=dataaugmentations,
                    experimenttracking=et)

            if et is not None:
                et.stop()

            log(ParameterResearchLoggerFields(name=trainresult.name,
                                              params=trainresult.params,
                                              mem_params=trainresult.mem_params,
                                              accuracy=trainresult.acc,
                                              tune_config=str(tune_config)))

            for postprocessing in config.get('postprocessing', []):
                ppp = {k: v for k,v in postprocessing.get('params', {}).items()} # Workaround tomlkit bug where some nested dict would lose their items
                trainresult = getattr(qualia.postprocessing, postprocessing['kind'])(**ppp)(
                    trainresult=trainresult,
                    model_conf=m)
                if trainresult.log:
                    log(ParameterResearchLoggerFields(name=trainresult.name,
                                                      params=trainresult.params,
                                                      mem_params=trainresult.mem_params,
                                                      accuracy=trainresult.acc,
                                                      tune_config=str(tune_config)))
                if postprocessing.get('export', False):
                    trainresult.framework.export(trainresult.model, f'{trainresult.name}_r{i}')

            return trainresult.acc



        if config['parameter_research'].get('study', {}).get('load', False):
            study = optuna.load_study(**config['parameter_research'].get('study', {}).get('params', {}))
        else:
            study = optuna.create_study(**config['parameter_research'].get('study', {}).get('params', {}))
        study.optimize(training_function, **config['parameter_research'].get('optimize', {}).get('params', {}))

        return loggers
