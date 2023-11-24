from __future__ import annotations

import dataclasses
import importlib.util
import logging
from dataclasses import dataclass

from qualia_core.typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType  # noqa: TCH003

logger = logging.getLogger(__name__)

@dataclass
class QualiaComponent:
    dataset: ModuleType | None = None
    deployment: ModuleType | None = None
    learningframework: ModuleType | None = None
    postprocessing: ModuleType | None = None
    preprocessing: ModuleType | None = None
    converter: ModuleType | None = None

    def package_names(self) -> tuple[str, ...]:
        return tuple(field.name for field in dataclasses.fields(self) if getattr(self, field.name) is not None)

def import_package_from_plugin(plugin_name: str, package_name: str) -> ModuleType | None:
    if importlib.util.find_spec(f'{plugin_name}.{package_name}') is None:
        logger.info('%s module not found in "%s" plugin', package_name, plugin_name)
        return None

    return importlib.import_module(f'{plugin_name}.{package_name}')

def load_plugin(plugin_name: str) -> QualiaComponent:

    packages = {package_name: import_package_from_plugin(plugin_name=plugin_name, package_name=package_name)
                    for package_name in ['dataset', 'deployment', 'preprocessing', 'learningframework', 'postprocessing']}

    component = QualiaComponent(**packages)
    component.converter = component.postprocessing

    logger.info("Loaded component '%s' with packages %s", plugin_name, component.package_names())

    return component

def load_plugins(plugin_names: list[str]) -> QualiaComponent:
    packages = load_plugin('qualia_core')
    for plugin_name in plugin_names:
        plugin = load_plugin(plugin_name)
        for package_name in plugin.package_names():
            getattr(packages, package_name).__dict__.update(getattr(plugin, package_name).__dict__)

    return packages
