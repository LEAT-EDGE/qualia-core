from __future__ import annotations

import importlib.util
import sys

from qualia_core.experimenttracking.NeptuneBase import NeptuneBase
from qualia_core.typing import TYPE_CHECKING, RecursiveConfigDict

from .ExperimentTrackingPyTorch import ExperimentTrackingPyTorch

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

if TYPE_CHECKING and importlib.util.find_spec('neptune'):
    from neptune.new.integrations.lightning.pytorch import NeptuneLogger  # noqa: TCH002


class Neptune(NeptuneBase, ExperimentTrackingPyTorch):
    def __init__(self, project_name: str, config_file: str='conf/neptune.toml') -> None:
        super().__init__(project_name=project_name, config_file=config_file)

    @override
    def start(self, name: str | None = None) -> None:
        from neptune.new.integrations.lightning.pytorch import NeptuneLogger

        project_name = f'{self.project_name}_{name}' if name is not None else self.project_name

        self.neptune_logger = NeptuneLogger(
            api_key=self.api_key,
            project=f'{self.project_namespace}/{project_name}',
            source_files=self.source_files,
            close_after_fit=False)

    @override
    def stop(self) -> None:
        self.neptune_logger.experiment.stop()

    @override
    def _hyperparameters(self, params: RecursiveConfigDict) -> None:
        for k, v in params.items():
            self.neptune_logger.experiment[k].log(v)

    @property
    def logger(self) -> NeptuneLogger:
        return self.neptune_logger

