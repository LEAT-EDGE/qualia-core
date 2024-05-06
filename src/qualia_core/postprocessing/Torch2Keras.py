from __future__ import annotations

import copy
import dataclasses
import logging
import re
import sys
from pathlib import Path
from typing import Any, Optional, Union, cast

import numpy.typing
import tomlkit
from torch import nn

try: # Keras 3.x
    from keras.src.ops.node import Node  # type: ignore[import-untyped]
except ModuleNotFoundError:
    from keras.src.engine.node import Node  # type: ignore[import-untyped]

from qualia_core.typing import TYPE_CHECKING, ModelConfigDict

from .PostProcessing import PostProcessing

if TYPE_CHECKING:
    import keras  #type: ignore[import-untyped] # noqa: TCH002
    from keras.layers import Layer  # type: ignore[import-untyped] # noqa: TCH002

    from qualia_core.learningframework.LearningFramework import LearningFramework  # noqa: TCH001
    from qualia_core.qualia import TrainResult  # noqa: TCH001

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)

RecursiveDictStr = dict[str, Union['RecursiveDictStr', str]]

class Torch2Keras(PostProcessing[nn.Module]):
    def __init__(self, mapping: str) -> None:
        super().__init__()
        with Path(mapping).open() as f:
            self.__mapping: RecursiveDictStr = tomlkit.parse(f.read())

    def __map_object_to_dict(self,
                             lookup_dict: RecursiveDictStr | str,
                             target_dict: nn.Module | list[nn.Module],
                             result_dict: dict[nn.Module, str] | None = None) -> dict[nn.Module, str]:
        """Get each layer of the PyTorch model according to the hierarchy defined in the 'mapping' parameter and create a flat dict
        with each layer as key and the name of the corresponding Keras layer as value (the leaves of the 'mapping' dict).
        """
        result_dict = result_dict if result_dict is not None else {}

        if not isinstance(lookup_dict, dict):
            if isinstance(target_dict, list):
                logger.error('Expected target_dict to be nn.Module when lookup_dict is str, but got list')
                raise TypeError
            return {target_dict: lookup_dict}

        for k, v in lookup_dict.items():
            if isinstance(target_dict, list):
                result_dict = {**self.__map_object_to_dict(v, target_dict[int(k)], result_dict),
                               **result_dict}
            else:
                target_value = cast(Optional[nn.Module], getattr(target_dict, k, None))

                if target_value is None:
                    logger.warning('Could not find %s in target_dict, ignoring', k)
                    continue

                result_dict = {
                        **self.__map_object_to_dict(v, target_value, result_dict),
                        **result_dict}
        return result_dict

    def __extract_weight_type_from_name(self, name: str) -> str | None:
        wt = re.search('^.+?/(.*):.+?$', name)
        if wt is None:
            return None
        return wt.group(1)

    def __reformat_fc_weights_data(self, tf_layers: list[Layer]) -> None:
        # After Flatten comes Dense, must reshape weights and swap axes for channels_first
        from tensorflow.keras.layers import Dense, Flatten  # type: ignore[import-untyped]
        for layer in tf_layers:
            if isinstance(layer, Flatten):
                # layer.outbound_nodes missing with Keras 3.x
                outnodes = cast(list[Node], layer.outbound_nodes
                                if hasattr(layer, 'outbound_nodes') else layer._outbound_nodes)
                for outnode in outnodes:
                    # Keras 3.x "layer" is called "operation"
                    dense = outnode.operation if hasattr(outnode, 'operation') else outnode.layer
                    if not isinstance(dense, Dense):
                        logger.error('Flatten layer must be followed by Dense')
                        raise TypeError

                    kernel = cast(numpy.typing.NDArray[Any], dense.kernel.numpy())
                    # Keras 3.x does not expose layer.input_shape directly
                    input_shape = cast(tuple[int, ...], layer.get_build_config()['input_shape']
                                       if not hasattr(layer, 'input_shape') else layer.input_shape)
                    units = cast(int, dense.units)

                    # reshape using Flatten input shape (for example last Conv output)
                    kernel = kernel.reshape(input_shape[-1:] + input_shape[1:-1] + (units, ))
                    kernel = kernel.swapaxes(0, 1)
                    kernel = kernel.reshape((-1, units))
                    dense.set_weights([kernel] + dense.get_weights()[1:])

    @override
    def __call__(self,
                 trainresult: TrainResult,
                 model_conf: ModelConfigDict) -> tuple[TrainResult, ModelConfigDict]:
        from keras.layers import Conv1D, Dense  # type: ignore[import-untyped]

        import qualia_core.learningmodel.keras as keras_learningmodels
        from qualia_core.learningframework import Keras

        framework = trainresult.framework
        model = trainresult.model
        model_nparams = framework.n_params(model)
        model.eval()

        # For activations_range, convert module name to identifier, '.' replaced with '_'
        # This is what QuantizationAwareTraining activations_range.txt uses
        from torch.fx.graph import _Namespace, _snake_case
        namespace = _Namespace()
        name_to_layer_torch = {namespace.create_name(_snake_case(name), m)
                                if name else name: m for name, m in model.named_modules()}

        framework = Keras()
        tf_params = copy.deepcopy(model_conf.get('params', {}))

        if 'quant_params' in tf_params:
            logger.warning('Dropping quant_params from model params, construction of quantized model is unsupported for Keras')
            del tf_params['quant_params']

        tf_model = getattr(keras_learningmodels, model_conf['kind'])(**tf_params)
        tf_model_nparams = framework.n_params(tf_model)

        framework.summary(tf_model)

        if model_nparams != tf_model_nparams:
            logger.error('Different number of parameters: PyTorch=%s, Keras=%s',
                         model_nparams,
                         tf_model_nparams)
            raise RuntimeError

        weights_name_tf_to_torch = { 'kernel': 'weight', 'bias': 'bias' }
        weights_transpose = {
            Conv1D: { 'kernel': (2, 1, 0), 'bias': (0,) },
            Dense: {'kernel': (1, 0), 'bias': (0,)},
        }

        layers_torch_to_tf = self.__map_object_to_dict(self.__mapping, model)
        layers_tf_to_torch = {v: k for k, v in layers_torch_to_tf.items()} # Reverse lookup

        for layer in tf_model.layers:
            if len(layer.weights) > 0: #layer need weights
                for weight in layer.weights:
                    # Keras 3.x 'name' does not contain layer name, 'path' correponds to older version 'name'
                    weight_type =  weight.name if hasattr(weight, 'path') else self.__extract_weight_type_from_name(weight.name)
                    if weight_type is None:
                        logger.error("Could not find weight type in name '%s'", weight.name)
                        raise RuntimeError
                    torch_layer = layers_tf_to_torch[layer.name]
                    weight_values_torch = getattr(torch_layer, weights_name_tf_to_torch[weight_type])
                    weight_values_np = weight_values_torch.data.cpu().numpy()

                    weight.assign(weight_values_np.transpose(weights_transpose[layer.__class__][weight_type]))

        self.__reformat_fc_weights_data(tf_model.layers)

        # Compile Keras model
        tf_model.compile(loss='categorical_crossentropy', metrics=['categorical_accuracy'])

        ### Convert Activations Range file with TF layer names
        activations_range_filepath = Path('out')/'learningmodel'/f'{trainresult.name}_r{trainresult.i}_activations_range.txt'
        if activations_range_filepath.is_file():
            activations_range: dict[str, str] = {}
            with activations_range_filepath.open() as f:
                for line in f:
                    r = line.strip().split(',', 1)
                    activations_range[r[0]] = r[1]
            with activations_range_filepath.with_suffix('.h5.txt').open('w') as f:
                for lname, arange in activations_range.items():
                    layer_torch = name_to_layer_torch[lname]
                    if layer_torch not in layers_torch_to_tf:
                        logger.warning('Unusued layer %s', lname)
                    else:
                        print(f'{layers_torch_to_tf[layer_torch]},{arange}', file=f)
        else:
            logger.info('No activation range file found, skipping conversion')

        new_model_conf = copy.deepcopy(model_conf)
        new_model_conf['params'] = tf_params

        return dataclasses.replace(trainresult, model=tf_model, framework=framework, log=False), new_model_conf

    @override
    def process_framework(self, framework: LearningFramework[nn.Module]) -> LearningFramework[keras.Model]:
        from qualia_core.learningframework.Keras import Keras
        return Keras()
