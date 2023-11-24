from __future__ import annotations

import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)


class TensorFlowInitializer:
    __tensorflow_initialized = False

    def set_seed(self, seed: int) -> None:
        # From https://www.reddit.com/r/MachineLearning/comments/ge32bi/how_to_reliably_compare_experiments_when_tuning/fpl4owx/
        import os
        import sys

        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        # If PYTHONHASHSEED is not set to 0 or CUBLAS_WORKSPACE_CONFIG not set to ':4096:8' (required for CuBlas determinism)
        # restart interpreter (env var has to be set before python starts)
        if os.environ.get('PYTHONHASHSEED', '') != str(seed) or os.environ.get('CUBLAS_WORKSPACE_CONFIG', '') != ':4096:8':
            os.execle(sys.executable,
                      sys.executable,
                      *sys.argv,
                      {'PYTHONHASHSEED': str(seed), 'CUBLAS_WORKSPACE_CONFIG': ':4096:8', **os.environ})

        import random
        random.seed(seed)

        import numpy as np
        np.random.seed(seed)  # noqa: NPY002 We do want to initialize the legacy global generator

        import qualia_core.random
        qualia_core.random.shared.seed(seed)

        try:
            import torch
            _ = torch.manual_seed(seed)  # type: ignore[untyped-def]

            import pytorch_lightning
            _ = pytorch_lightning.seed_everything(seed)
        except ImportError:
            logger.warning('PyTorch not loaded and not seeded')

        try:
            import tensorflow as tf  # type: ignore[import-untyped]
            tf.random.set_seed(seed)  # type: ignore[untyped-def]
        except ImportError:
            logger.warning('TensorFlow not loaded and not seeded')

    def __call__(self,
                 reserve_gpu: bool = True,  # noqa: FBT001, FBT002
                 gpu_memory_growth: bool = True,  # noqa: FBT001, FBT002
                 debug: bool = False,  # noqa: FBT001, FBT002
                 seed: int | None = None) -> None:
        if TensorFlowInitializer.__tensorflow_initialized:
            logger.warning('TensorFlow already initialized by TensorFlowInitialize, new settings not be applied properly.')
        elif 'tensorflow' in sys.modules:
            logger.error('TensorFlow already imported previously. TensorFlowInitializer must be called first!')
            raise RuntimeError

        if not debug:
            import os
            # Needs to be done before the first tensorflow import ever
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

        if seed is not None:
            self.set_seed(seed)

        try:
            import tensorflow as tf  # type: ignore[import-untyped]
        except ImportError:
            logger.warning('TensorFlow not loaded, not setting up devices')
            return

        if debug:
            tf.debugging.set_log_device_placement(enabled=True)  # type: ignore[untyped-def]

        if not reserve_gpu:
            # Hide GPUs when not training to not allocate any resource
            tf.config.set_visible_devices([], 'GPU')  # type: ignore[untyped-def]
        else:
            # From https://www.tensorflow.org/guide/gpu
            # Limiting GPU memory growth
            gpus: list[Any] = tf.config.experimental.list_physical_devices('GPU')  # type: ignore[untyped-def]
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, gpu_memory_growth)  # type: ignore[untyped-def]

            logical_gpus: list[Any] = tf.config.experimental.list_logical_devices('GPU')  # type: ignore[untyped-def]
            logger.info('%d Physical GPUs, %d Logical GPUs', len(gpus), len(logical_gpus))

        TensorFlowInitializer.__tensorflow_initialized = True
