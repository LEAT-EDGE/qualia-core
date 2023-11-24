from pathlib import Path

import tomlkit

from .ExperimentTracking import ExperimentTracking


class NeptuneBase(ExperimentTracking):
    project_name: str
    api_key: str
    project_namespace: str
    source_files: list[str]

    def __init__(self, project_name: str, config_file: str='conf/neptune.toml') -> None:
        super().__init__()

        self.project_name = project_name

        with Path(config_file).open() as f:
            config_dict = tomlkit.parse(f.read())

            self.api_key = config_dict['neptune']['api_key']
            self.project_namespace = config_dict['neptune']['project_namespace']
            self.source_files = config_dict['neptune']['source_files']
