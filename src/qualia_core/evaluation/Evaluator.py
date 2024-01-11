from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from qualia_core import random
from qualia_core.typing import TYPE_CHECKING

# We are inside a TYPE_CHECKING block but our custom TYPE_CHECKING constant triggers TCH001-TCH003 so ignore them
if TYPE_CHECKING:
    import numpy as np  # noqa: TCH002
    import numpy.typing  # noqa: TCH002

    from qualia_core.dataaugmentation.DataAugmentation import DataAugmentation  # noqa: TCH001
    from qualia_core.datamodel.RawDataModel import RawDataModel  # noqa: TCH001
    from qualia_core.evaluation.Stats import Stats  # noqa: TCH001
    from qualia_core.learningframework.LearningFramework import LearningFramework  # noqa: TCH001

logger = logging.getLogger(__name__)

class Evaluator(ABC):
    def apply_dataaugmentations(self,
                                framework: LearningFramework[Any],
                                dataaugmentations: list[DataAugmentation] | None,
                                test_x: numpy.typing.NDArray[np.float32],
                                test_y: numpy.typing.NDArray[np.int32]) -> tuple[numpy.typing.NDArray[np.float32],
                                                                                 numpy.typing.NDArray[np.int32]]:
        """Apply evaluation :class:`qualia_core.dataaugmentation.DataAugmentation.DataAugmentation` to dataset.

        Only the :class:`qualia_core.dataaugmentation.DataAugmentation.DataAugmentation`
        with :attr:`qualia_core.dataaugmentation.DataAugmentation.DataAugmentation.evaluate` set are applied.
        This should not be used to apply actual data augmentation to the data, but rather use the conversion or transform
        :class:`qualia_core.dataaugmentation.DataAugmentation.DataAugmentation` modules.

        :param framework: The :class:`qualia_core.learningframework.LearningFramework.LearningFramework` instance
            providing the :meth:`qualia_core.learningframework.LearningFramework.LearningFramework.apply_dataaugmentation` method
            compatible with the provided :class:`qualia_core.dataaugmentation.DataAugmentation.DataAugmentation`
        :param dataaugmentations: List of :class:`qualia_core.dataaugmentation.DataAugmentation.DataAugmentation` objects
        :param test_x: Input data to apply data augmentation to
        :param test_y: Input labels to apply data augmentation to
        :return: Tuple of data and labels after
            applying :class:`qualia_core.dataaugmentation.DataAugmentation.DataAugmentation` sequentially
        """
        if dataaugmentations is None:
            return test_x, test_y
        for da in dataaugmentations:
            if da.evaluate:
                test_x, test_y = framework.apply_dataaugmentation(da, test_x, test_y)
        return test_x, test_y

    def shuffle_dataset(self,
                        test_x: numpy.typing.NDArray[np.float32],
                        test_y: numpy.typing.NDArray[np.int32]) -> tuple[numpy.typing.NDArray[np.float32],
                                                                         numpy.typing.NDArray[np.int32]]:
        """Shuffle the input data, keeping the labels in the same order as the shuffled data.

        Shuffling uses the seeded shared random generator from :obj:`qualia_core.random.shared`.

        :param test_x: Input data
        :param test_y: Input labels
        :return: Tuple of shuffled data and labels
        """
        perms = random.shared.generator.permutation(len(test_y))
        test_x = test_x[perms]
        test_y = test_y[perms]
        return test_x, test_y

    def limit_dataset(self,
                      test_x: numpy.typing.NDArray[np.float32],
                      test_y: numpy.typing.NDArray[np.int32],
                      limit: int | None) -> tuple[numpy.typing.NDArray[np.float32],
                                                  numpy.typing.NDArray[np.int32]]:
        """Truncate dataset to ``limit`` samples.

        :param test_x: Input data
        :param test_y: Input labels
        :param limit: Number of samples to truncate to, data is returned as-is if `None` or 0
        :return: Tuple of data and labels limited to ``limit`` samples
        """
        if limit:
            test_x = test_x[:limit]
            test_y = test_y[:limit]
        return test_x, test_y

    def compute_accuracy(self,
                         preds: list[int],
                         truth: numpy.typing.NDArray[np.int32]) -> float:
        """Compute accuracy from the target results.

        :param results: List of :class:`Result` from inference on the target
        :param test_y: Array of ground truth one-hot encoded
        :return: Accuracy (micro) between 0 and 1
        """
        correct = 0
        for line, pred in zip(truth, preds):
            logger.debug('%s %s', line.argmax(), pred)
            if line.argmax() == pred:
                correct += 1

        return correct / len(preds)

    @abstractmethod
    def evaluate(self,  # noqa: PLR0913
                 framework: LearningFramework[Any],
                 model_kind: str,
                 dataset: RawDataModel,
                 target: str,
                 tag: str,
                 limit: int | None = None,
                 dataaugmentations: list[DataAugmentation] | None = None) -> Stats | None:
        ...
