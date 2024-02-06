from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from qualia_core.experimenttracking.ExperimentTracking import ExperimentTracking
from qualia_core.utils.file.DirectoryReader import DirectoryReader

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

if TYPE_CHECKING:
    from qualia_core.typing import RecursiveConfigDict

logger = logging.getLogger(__name__)

class ClearML(ExperimentTracking):
    def __init__(self,
                 project_name: str,
                 task_name: str,
                 sources_path: str | None = None,
                 ignores: list[str] | None = None,
                 offline_mode: bool = False) -> None:
        from clearml import Task

        super().__init__()
        self.__project_name = project_name
        self.__task_name = task_name
        self.__task: Task | None = None

        # Path to the current project sources, used as the "script path" and to find the version control repository (e.g. git)
        # By default use the "src/qualia" path under the current working directory
        src_path = Path(sources_path) if sources_path is not None else Path.cwd() / 'src' / 'qualia'

        # Paths to ignore when looking for requirements
        # Parsing everything would take much longer than necessary
        # This mostly leaves the src/ and tests/ directories (and conf)
        ignores_list = ['__pypackages__',
                        'logs',
                        'out',
                        'lightning_logs',
                        'build',
                        'third_party',
                        '.venv',
                        '.mypy_cache',
                        '.ruff_cache',
                        '.pyre',
                        '.pytest_cache',
                        'data',
                        'src/qualia/assets']
        if ignores is not None:
            ignores_list += ignores

        if offline_mode:
            Task.set_offline(offline_mode=offline_mode)

        self.__patch_clearml(extra_ignores=ignores_list, src_path=src_path)

    def __patch_clearml(self, extra_ignores: list[str], src_path: Path) -> None:
        """Monkey-patch ClearML methods for our use-case.

        ScriptInfo.get() patched to set the desired 'script_path' to find the git repository.
        pigar.GenerateReqs.__init__() patched to add more directories to the ignore list of get_requirements().
        """
        from clearml.backend_interface.task.repo import scriptinfo
        getScriptInfo = scriptinfo.ScriptInfo.get.__func__

        def getScriptInfoPatched(cls, filepaths=None, *args, **kwargs):
            return getScriptInfo(cls, *args, filepaths=[src_path], **kwargs)

        scriptinfo.ScriptInfo.get = classmethod(getScriptInfoPatched)

        import clearml.utilities.pigar.__main__ as pigar
        generateReqsInit = pigar.GenerateReqs.__init__
        def generateReqsInitPatched(self: pigar.GenerateReqs, *, ignores: list[str], **kwargs) -> None:
            generateReqsInit(self, ignores=ignores + extra_ignores, **kwargs)
        pigar.GenerateReqs.__init__ = generateReqsInitPatched


    @override
    def start(self, name: str | None = None) -> None:
        from clearml import Task
        if Task.current_task() is None:
            task_name = f'{self.__task_name}_{name}' if name is not None else self.__task_name
            self.__task = Task.init(project_name=self.__project_name, task_name=task_name, reuse_last_task_id=False)

    @override
    def stop(self) -> None:
        if self.__task is not None:
            self.__task.close()
        self.__task = None

    @override
    def _hyperparameters(self, params: RecursiveConfigDict) -> None:
        if self.__task is not None:
            self.__task.connect(params)

    @override
    @classmethod
    def initializer(cls) -> None:
        """Connect to task in PyTorch Lightning Trainer subprocess with e.g. ddp_spawn."""
        from clearml import Task
        Task.current_task()

    @classmethod
    def import_and_clear_all_offline_sessions(cls) -> None:
        from clearml import Task

        from qualia_core.utils.logger.setup_root_logger import setup_root_logger

        # We main not be called from qualia_core.main:main so always setup logging to show logger.info()
        setup_root_logger(colored=True)

        offline_cache_path = Path(sys.argv[1]) if len(sys.argv) > 0 else Path.home()/'.clearml'/'cache'/'offline'

        dr = DirectoryReader()
        sessions = list(dr.read(directory=offline_cache_path, ext='.zip', recursive=False))
        imported_sessions: list[str] = []
        logger.info('%d session%s to import.', len(sessions), 's' if len(sessions) > 1 else '')

        for f in sessions:
            imported_session = Task.import_offline_session(session_folder_zip=str(f))
            if imported_session is None:
                logger.error('Failed to import session %s', str(f))
            else:
                imported_sessions.append(imported_session)

            logger.info('Deleting session archive %s', str(f))
            f.unlink()

        logger.info('Imported %d session%s.', len(imported_sessions), 's' if len(imported_sessions) > 1 else '')

    @property
    def logger(self) -> None:
        return None
