from __future__ import annotations

import logging
from importlib.resources import files
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from qualia_core.typing import TYPE_CHECKING, ConfigDict
from qualia_core.utils.merge_dict import merge_dict
from qualia_core.utils.path import lookup_file, resources_to_path

if TYPE_CHECKING:
    from qualia_core.typing import RecursiveConfigDict

logger = logging.getLogger(__name__)


def validate_config_dict(config: RecursiveConfigDict) -> ConfigDict | None:
    ta = TypeAdapter(ConfigDict)
    try:
        validated_config: ConfigDict = ta.validate_python(config, strict=False)

        extra_dict_a_keys = [k for k in config if k not in validated_config]
        extra_dict_b_keys = [k for k in validated_config if k not in config]
        non_matching_values = {k: v for k, v in validated_config.items() if k in config and v != config[k]}

        if extra_dict_a_keys:
            logger.error('Missing keys after validation: %s', extra_dict_a_keys)

        if extra_dict_b_keys:
            logger.error('Extra keys after validation: %s', extra_dict_b_keys)

        if non_matching_values:
            for k, v in non_matching_values.items():
                logger.error('Different value after validation for key: %s', k)
                logger.error('Before validation:\n%s', config[k])
                logger.error('After validation:\n%s', v)

        if extra_dict_a_keys or extra_dict_b_keys or non_matching_values:
            return None

    except ValidationError:
        logger.exception('Error when validating configuration.')
        return None

    return validated_config


def parse_config(path: Path) -> tuple[RecursiveConfigDict, str]:
    import tomlkit

    with path.open() as f:
        toml_config = tomlkit.parse(f.read())

    # Convert to built-in Python types
    config: RecursiveConfigDict = toml_config.unwrap()

    return config, path.stem


def merge_model_template(config: RecursiveConfigDict) -> RecursiveConfigDict:
    # Merge settings from template into individual models
    if 'model_template' in config:
        models = config['model']
        model_template = config['model_template']

        if not isinstance(models, list):
            logger.error('`model` must be a list, got: %s', type(models))
            raise TypeError

        if not isinstance(model_template, dict):
            logger.error('`model_template` must be a dict, got: %s', type(model_template))
            raise TypeError

        for i, model in enumerate(models):
            if not isinstance(model, dict):
                logger.error('`model[%d]` must be a dict, got: %s', i, type(model))
                raise TypeError
            models[i] = merge_dict(model, model_template)

    return config


def load_config(path: Path, args: RecursiveConfigDict | None = None) -> tuple[ConfigDict | None, str]:
    # Parse config file
    config, configname = parse_config(path)
    # Overwrite config file params with command line arguments
    config_overwritten = merge_dict(args, config, merge_lists=True) if args is not None else config

    # Default include file search path
    # First path takes precedence
    # - Search conf subdir inside the current directory
    # - Search conf directory of qualia-core, if installed as editable
    # - Search conf directory of all plugins, if installed as editable
    include_search_paths = [Path('conf'),
                            resources_to_path(files('qualia_core')).parent.parent / 'conf',
                            *[resources_to_path(files(p)).parent.parent / 'conf'
                                for p in config_overwritten['bench'].get('plugins', [])]]

    # Prepend paths specified in config file or command line args to search path
    additional_include_search_paths = [Path(path) for path in config_overwritten.get('include_search_paths', [])]
    include_search_paths = additional_include_search_paths + include_search_paths

    # Load include files
    includes: list[str] = config_overwritten.get('includes', [])
    while includes:
        filename = Path(includes.pop(0))
        file_path = lookup_file(search_paths=include_search_paths, filename=filename)
        if file_path:
            logger.info('Including "%s"', file_path)
            config_include, _ = parse_config(file_path)
            # Included file may include additional files, add them to includes list
            includes += config_include.get('includes', [])
            # Main config file and command line args take precedence over included file
            config_overwritten = merge_dict(config_overwritten, config_include, merge_lists=True)
        else:
            logger.warning('Include file "%s" not found', filename)
            logger.warning('Search paths: %s', include_search_paths)

    config_overwritten = merge_model_template(config_overwritten)

    validated_config = validate_config_dict(config_overwritten)

    return validated_config, configname
