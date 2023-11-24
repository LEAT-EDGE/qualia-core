import dataclasses
import sys

import keras  # type: ignore[import-untyped]

from qualia_core.qualia import TrainResult
from qualia_core.typing import ModelConfigDict

from .PostProcessing import PostProcessing

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class RemoveKerasSoftmax(PostProcessing[keras.Model]):
    @override
    def __call__(self, trainresult: TrainResult, model_conf: ModelConfigDict) -> tuple[TrainResult, ModelConfigDict]:
        from keras.activations import softmax  # type: ignore[import-untyped]
        from keras.layers import Activation  # type: ignore[import-untyped]

        model = trainresult.model
        assert isinstance(model.layers[-1], Activation)
        assert model.layers[-1].activation == softmax

        require_compile = model._is_compiled
        if require_compile:
            compile_params = {'loss': model.loss, 'optimizer': model.optimizer, 'metrics': model.compiled_metrics.metrics}

        # Remove softmax
        model = keras.Model(model.input, model.layers[-2].output, name=model.name)

        if require_compile:
            model.compile(**compile_params)

        model.summary()
        return dataclasses.replace(trainresult, model=model), model_conf
