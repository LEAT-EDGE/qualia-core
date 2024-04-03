#!/usr/bin/env python3

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import colorful as cf  # type: ignore[import-untyped]

import qualia_core.utils.args
import qualia_core.utils.config
import qualia_core.utils.plugin
from qualia_core.typing import TYPE_CHECKING
from qualia_core.utils.logger import Logger
from qualia_core.utils.logger.setup_root_logger import setup_root_logger
from qualia_core.utils.merge_dict import merge_dict

if TYPE_CHECKING:
    from types import ModuleType  # noqa: TCH003

    from qualia_core.postprocessing.Converter import Converter  # noqa: TCH001
    from qualia_core.typing import ConfigDict

logger = logging.getLogger(__name__)

def qualia(action: str,
           config: ConfigDict,
           configname: str) -> dict[str, list[Any]]:
    # Must be called first to configure devices before initializing Keras
    from qualia_core.utils import TensorFlowInitializer
    tfi = TensorFlowInitializer()
    tfi(seed=config['bench']['seed'],
        reserve_gpu=(config['bench'].get('gpus', None) and 'train' in sys.argv[2:]) or config['bench'].get('reserve_gpu', False),
        gpu_memory_growth=config['bench'].get('gpu_memory_growth', True))

    from qualia_core import command
    from qualia_core.utils import Git

    qualia = qualia_core.utils.plugin.load_plugins(config['bench'].get('plugins', []))

    git = Git()
    Logger.logpath /= config['bench']['name'] # Store logfiles in separate directory according to the config bench name
    Logger.prefix = f'{git.short_hash()}_' # Add prefix according to current git commit
    loggers: dict[str, Logger] = {} # Keep track of the loggers we used to return them

    learningframework = getattr(qualia.learningframework,
                                config['learningframework']['kind'])(**config['learningframework'].get('params', {}))

    dataset = getattr(qualia.dataset, config['dataset']['kind'])(**config['dataset'].get('params', {}))
    converter: type[Converter] = getattr(qualia.converter, config.get('deploy', {}).get('converter', {}).get('kind', ''), None)
    if config.get('deploy', {}).get('converter', '') and not converter:
        logger.error("Converter '%s' not found", config['deploy']['converter'])
        raise ValueError

    deployers: ModuleType = getattr(qualia.deployment, config.get('deploy', {}).get('deployer', {}).get('kind', ''), None)
    if not deployers and converter and converter.deployers is not None:
        deployers = converter.deployers

    dataaugmentations = [getattr(learningframework.dataaugmentations, da['kind'])(**da.get('params', {}))
                             for da in config.get('data_augmentation', {})]

    if action == 'preprocess_data':
        preprocess_data_command = command.PreprocessData()
        loggers.update(preprocess_data_command(qualia=qualia,
                                               dataset=dataset,
                                               config=config))
        return {k: v.content for k, v in loggers.items()}

    dataset_pipeline = dataset
    for preprocessing in config.get('preprocessing', []):
        dataset_pipeline = getattr(qualia.preprocessing,
                                   preprocessing['kind'])(**preprocessing.get('params', {})).import_data(dataset_pipeline)

    data = dataset_pipeline.import_data()

    if action == 'train':
        train_command = command.Train()
        loggers.update(train_command(qualia=qualia,
                                     learningframework=learningframework,
                                     dataaugmentations=dataaugmentations,
                                     data=data,
                                     config=config,
                                     git=git))
    elif action == 'prepare_deploy':
        prepare_deploy_command = command.PrepareDeploy()
        loggers.update(prepare_deploy_command(qualia=qualia,
                                              learningframework=learningframework,
                                              converter=converter,
                                              deployers=deployers,
                                              data=data,
                                              config=config,
                                              git=git))

    elif action == 'deploy_and_evaluate':
        deploy_and_evaluate_command = command.DeployAndEvaluate()
        loggers.update(deploy_and_evaluate_command(qualia=qualia,
                                                   learningframework=learningframework,
                                                   dataaugmentations=dataaugmentations,
                                                   converter=converter,
                                                   deployers=deployers,
                                                   data=data,
                                                   config=config,
                                                   git=git))
    elif action == 'parameter_research':
        parameter_research = command.ParameterResearch()
        loggers.update(parameter_research(qualia=qualia,
                                          learningframework=learningframework,
                                          dataaugmentations=dataaugmentations,
                                          data=data,
                                          config=config,
                                          git=git))
    else:
        logger.error('Invalid action: %s', action)
        raise ValueError

    return {k: v.content for k, v in loggers.items()}


def main() -> int:
    cf.use_style('solarized')  # type: ignore[untyped-def]

    if len(sys.argv) < 3:
        print(f'Usage: {sys.argv[0]} <config.toml> <preprocess_data|train|prepare_deploy|deploy_and_evaluate|parameter_research> [config_params...]')
        sys.exit(1)

    setup_root_logger(colored=True)

    config, configname = qualia_core.utils.config.parse_config(Path(sys.argv[1]))
    config_overwrite = qualia_core.utils.args.parse_args(sys.argv[3:])
    # Overwrite config file params with command line arguments
    config_overwritten = merge_dict(config_overwrite, config, merge_lists=True)
    validated_config = qualia_core.utils.config.validate_config_dict(config_overwritten)
    if validated_config is None:
        logger.error('Could not load configuration.')
        return 1

    loggers = qualia(sys.argv[2], config=validated_config, configname=configname)
    logger.info('%s', loggers)
    return 0

if __name__ == '__main__':
    sys.exit(main())
