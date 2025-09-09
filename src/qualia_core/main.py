#!/usr/bin/env python3

from __future__ import annotations

import logging
import sys
from importlib.resources import files
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
from qualia_core.utils.path import lookup_file, resources_to_path

if TYPE_CHECKING:
    from types import ModuleType

    from qualia_core.postprocessing.Converter import Converter
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
    if git.short_hash:
        Logger.prefix = f'{git.short_hash}_' # Add prefix according to current git commit
    else:
        logger.info('Local git repository not found, last commit hash will not be prepended to log file names')

    loggers: dict[str, Logger[Any]] = {} # Keep track of the loggers we used to return them

    learningframework = getattr(qualia.learningframework,
                                config['learningframework']['kind'])(**config['learningframework'].get('params', {}))

    dataset = getattr(qualia.dataset, config['dataset']['kind'])(**config['dataset'].get('params', {}))
    converter: type[Converter] = getattr(qualia.converter, config.get('deploy', {}).get('converter', {}).get('kind', ''), None)
    if config.get('deploy', {}).get('converter', {}).get('kind') and not converter:
        logger.error("Converter '%s' not found", config['deploy']['converter']['kind'])
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
                                     config=config))
    elif action == 'prepare_deploy':
        prepare_deploy_command = command.PrepareDeploy()
        loggers.update(prepare_deploy_command(qualia=qualia,
                                              learningframework=learningframework,
                                              converter=converter,
                                              deployers=deployers,
                                              data=data,
                                              config=config))

    elif action == 'deploy_and_evaluate':
        deploy_and_evaluate_command = command.DeployAndEvaluate()
        loggers.update(deploy_and_evaluate_command(qualia=qualia,
                                                   learningframework=learningframework,
                                                   dataaugmentations=dataaugmentations,
                                                   converter=converter,
                                                   deployers=deployers,
                                                   data=data,
                                                   config=config))
    elif action == 'evaluate':
        evaluate_command = command.Evaluate()
        loggers.update(evaluate_command(qualia=qualia,
                                        learningframework=learningframework,
                                        dataaugmentations=dataaugmentations,
                                        converter=converter,
                                        deployers=deployers,
                                        data=data,
                                        config=config))
    elif action == 'parameter_research':
        parameter_research = command.ParameterResearch()
        loggers.update(parameter_research(qualia=qualia,
                                          learningframework=learningframework,
                                          dataaugmentations=dataaugmentations,
                                          data=data,
                                          config=config))
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

    # Parse config file
    config, configname = qualia_core.utils.config.parse_config(Path(sys.argv[1]))
    # Parse command line args
    config_overwrite = qualia_core.utils.args.parse_args(sys.argv[3:])
    # Overwrite config file params with command line arguments
    config_overwritten = merge_dict(config_overwrite, config, merge_lists=True)

    # Default include file search path
    # First path takes precedence
    # - Search conf subdir inside the current directory
    # - Search conf directory of qualia-core, if installed as editable
    # - Search conf directory of all plugins, if installed as editable
    include_search_paths = [Path('conf'),
                            resources_to_path(files('qualia_core')).parent.parent/'conf',
                            *[resources_to_path(files(p)).parent.parent/'conf'
                                for p in config['bench'].get('plugins', [])]]

    # Prepend paths specified in config file or command line args to search path
    additional_include_search_paths = [Path(path) for path in config_overwritten.get('include_search_paths', [])]
    include_search_paths = additional_include_search_paths + include_search_paths

    # Load include files
    for filename in config_overwritten.get('includes', []):
        file_path = lookup_file(search_paths=include_search_paths, filename=filename)
        if file_path:
            logger.info('Including "%s"', file_path)
            config_include, _ = qualia_core.utils.config.parse_config(file_path)
            # Main config file and command line args take precedence over included file
            config_overwritten = merge_dict(config_overwritten, config_include, merge_lists=True)
        else:
            logger.warning('Include file "%s" not found', filename)
            logger.warning('Search paths: %s', include_search_paths)

    validated_config = qualia_core.utils.config.validate_config_dict(config_overwritten)
    if validated_config is None:
        logger.error('Could not load configuration.')
        return 1

    loggers = qualia(sys.argv[2], config=validated_config, configname=configname)
    logger.info('%s', loggers)
    return 0

if __name__ == '__main__':
    sys.exit(main())
