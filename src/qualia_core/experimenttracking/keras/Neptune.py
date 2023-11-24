import sys
from typing import TYPE_CHECKING, Optional

from qualia_core.experimenttracking.NeptuneBase import NeptuneBase
from qualia_core.typing import RecursiveConfigDict

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

if TYPE_CHECKING:
    from neptune.new.integrations.tensorflow_keras import NeptuneCallback

class Neptune(NeptuneBase):
    def __init__(self, project_name: str, config_file: str='conf/neptune.toml') -> None:
        super().__init__(project_name=project_name, config_file=config_file)

    @override
    def start(self, name: Optional[str] = None) -> None:
        import neptune.new as neptune
        from neptune.new.integrations.tensorflow_keras import NeptuneCallback

        project_name = f'{self.project_name}_{name}' if name is not None else self.project_name

        self.run = neptune.init(
            api_key=self.api_key,
            project_name=f'{self.project_namespace}/{project_name}',
            upload_source_files=self.source_files)
        self.neptune_cbk = NeptuneCallback(run=self.run, base_namespace='metrics')

    @override
    def stop(self) -> None:
        self.run.stop()

    @property
    def callback(self) -> 'NeptuneCallback':
        return self.neptune_cbk

    @override
    def _hyperparameters(self, params: RecursiveConfigDict) -> None:
        self.neptune_cbk.log_hyperparams(params)
