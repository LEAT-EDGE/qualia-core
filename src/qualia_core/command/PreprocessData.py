from __future__ import annotations

import logging
from typing import Any, Callable

from qualia_core.typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qualia_core.datamodel.DataModel import DataModel  # noqa: TCH001
    from qualia_core.typing import ConfigDict
    from qualia_core.utils.logger import Logger  # noqa: TCH001
    from qualia_core.utils.plugin import QualiaComponent  # noqa: TCH001

logger = logging.getLogger(__name__)

class PreprocessData:
    def __call__(self,
                 qualia: QualiaComponent,
                 dataset: Callable[[], DataModel[Any]],
                 config: ConfigDict) -> dict[str, Logger[Any]]:

        data = dataset()
        for preprocessing in config.get('preprocessing', []):
            data = getattr(qualia.preprocessing, preprocessing['kind'])(**preprocessing.get('params', {}))(data)
        logger.info('Exporting data')
        _ = data.export()

        return {}
