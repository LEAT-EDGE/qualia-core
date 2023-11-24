from __future__ import annotations

import logging

from pydantic import TypeAdapter, ValidationError

from qualia_core.typing import TYPE_CHECKING, ConfigDict
from qualia_core.utils.merge_dict import merge_dict

if TYPE_CHECKING:
    from pathlib import Path  # noqa: TCH003

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

    return config, path.stem

