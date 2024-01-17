from __future__ import annotations

import copy
import importlib.util
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, cast

import qualia_core.deployment.qualia_codegen

from .Converter import Converter

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

if TYPE_CHECKING:
    from torch import nn  # noqa: I001 # torch must be imported before keras to avoid deadlock
    import keras  # type: ignore[import] # No stubs for keras package
    import numpy.typing
    from qualia_codegen_core.graph import ModelGraph
    from qualia_codegen_core.graph.layers import TBaseLayer

    from qualia_core.learningframework.LearningFramework import LearningFramework

logger = logging.getLogger(__name__)

class QualiaCodeGen(Converter[Any]):
    deployers = qualia_core.deployment.qualia_codegen #: Suggested deployers

    __number_type: type[int | float]
    _h: str | None = None
    _name: str | None = None

    def __init__(self,
                 quantize: str,
                 long_width: int | None = None,
                 outdir: str | None = None) -> None:
        super().__init__()

        self.__quantize = quantize
        self.__outdir = Path(outdir) if outdir is not None else Path('out')/'qualia_codegen'

        if quantize == 'float32':
            self.__number_type = float
            self.__width = 32
            self.__long_width = 32 if long_width is None else long_width
        elif quantize == 'int16':
            self.__number_type = int
            self.__width = 16
            self.__long_width = 32 if long_width is None else long_width
        elif quantize == 'int8':
            self.__number_type = int
            self.__width = 8
            self.__long_width = 16 if long_width is None else long_width
        else:
            logger.error('Qualia-CodeGen only supports no (float32) quantization, int8 or int16 quantization, got %s', quantize)
            raise ValueError

    def convert_model_to_modelgraph(self, model: nn.Module | keras.Model) -> ModelGraph | None:
        from qualia_codegen_core.graph.layers import TAddLayer, TSumLayer

        modelgraph: ModelGraph | None = None

        if importlib.util.find_spec('torch') is None:
            logger.warning('Cannot find PyTorch, PyTorch support for Qualia-CodeGen will be unavailable')
        else:
            from torch import nn
            if isinstance(model, nn.Module):
                from qualia_codegen_core.graph import TorchModelGraph

                from qualia_core.learningmodel.pytorch.layers import Add, GlobalSumPool1d, GlobalSumPool2d
                custom_layers: dict[type[nn.Module], Callable[[nn.Module, TBaseLayer], tuple[type[TBaseLayer], list[Any]]]] = {
                        Add: lambda *_: (TAddLayer, []),
                        GlobalSumPool1d: lambda *_: (TSumLayer, [(-1,)]),
                        GlobalSumPool2d: lambda *_: (TSumLayer, [(-2, -1)]),
                        }
                modelgraph = TorchModelGraph(model).convert(custom_layers=custom_layers)

        if importlib.util.find_spec('keras') is None:
            logger.warning('Cannot find Keras, Keras support for Qualia-CodeGen will be unavailable')
        else:
            import keras  # type: ignore[import] # No stubs for keras package
            if isinstance(model, keras.Model):
                from qualia_codegen_core.graph import KerasModelGraph
                modelgraph = KerasModelGraph(model).convert()

        return modelgraph

    def convert_modelgraph_to_c(self, modelgraph: ModelGraph, output_path: Path) -> str | None:
        from qualia_codegen_core import Converter
        converter = Converter(output_path=output_path)
        return converter.convert_model(modelgraph)

    @override
    def convert(self,
                framework: LearningFramework[nn.Module | keras.Model],
                model: nn.Module | keras.Model,
                model_name: str,
                representative_dataset: numpy.typing.NDArray[Any]) -> QualiaCodeGen | None:
        from qualia_codegen_core.graph import Quantization
        from qualia_codegen_core.graph.ActivationsRange import ActivationsRange
        from qualia_codegen_core.graph.RoundMode import RoundMode

        self._name = f'{model_name}_q{self.__quantize}'

        framework.summary(model)

        modelgraph = self.convert_model_to_modelgraph(model)
        if modelgraph is None:
            logger.error('Could not convert model to ModelGraph')
            return None

        if self.__number_type == int: # Activation range only when using fixed-point quantization
            activations_range = ActivationsRange()

            if importlib.util.find_spec('keras') is not None:
                from qualia_codegen_core.graph import KerasModelGraph
                if isinstance(modelgraph, KerasModelGraph):
                    activations_range = activations_range.load(
                                            Path('out')/'learningmodel'/f'{model_name}_activations_range.h5.txt',
                                            cast(str, model.layers[0].name))

            if importlib.util.find_spec('torch') is not None:
                from qualia_codegen_core.graph import TorchModelGraph
                if isinstance(modelgraph, TorchModelGraph):
                    activations_range = activations_range.load(
                                            Path('out')/'learningmodel'/f'{model_name}_activations_range.txt',
                                            'input')

            # Populate quantization information for all layers from activations_range
            for node in modelgraph.nodes:
                if node.layer.name in activations_range:
                    node.q = Quantization(
                            number_type=self.__number_type,
                            width=self.__width,
                            long_width=self.__long_width,
                            weights_scale_factor=activations_range[node.layer.name].weights_q,
                            bias_scale_factor=activations_range[node.layer.name].bias_q,
                            output_scale_factor=activations_range[node.layer.name].activation_q,
                            weights_round_mode=activations_range[node.layer.name].weights_round_mode,
                            output_round_mode=activations_range[node.layer.name].activation_round_mode,
                            )
                else:
                    if not node.innodes:
                        logger.error('No quantization information for %s, and no previous layer to copy from.', node.layer.name)
                        return self
                    logger.warning('No quantization information for %s, applying first previous layer %s information',
                                   node.layer.name,
                                   node.innodes[0].layer.name)
                    node.q = copy.deepcopy(node.innodes[0].q)
        else:
            for node in modelgraph.nodes:
                # No scale factor if not fixed-point quantization on integers
                node.q = Quantization(
                        number_type=self.__number_type,
                        width=self.__width,
                        long_width=self.__long_width,
                        weights_scale_factor=0,
                        bias_scale_factor=None,
                        output_scale_factor=0,
                        weights_round_mode=RoundMode.NONE,
                        output_round_mode=RoundMode.NONE,
                        )
        # self.directory cannot be None as long as we define self._name above
        self._h = self.convert_modelgraph_to_c(modelgraph, output_path=cast(Path, self.directory))

        if self._h is None:
            logger.error('Could not convert ModelGraph to C')
            return None

        with (self.__outdir/self._name/'full_model.h').open('w') as f:
            _ = f.write(self._h)

        return self

    @property
    def h(self) -> str | None:
        return self._h

    @property
    def name(self) -> str | None:
        return self._name

    @property
    def directory(self) -> Path | None:
        if self.name is None:
            return None
        return self.__outdir / self.name

    @override
    def process_mem_params(self, mem_params: int) -> Callable[[LearningFramework[nn.Module | keras.Model],
                                                               nn.Module | keras.Model],
                                                              int]:
        def f(framework: LearningFramework[nn.Module | keras.Model], model: nn.Module | keras.Model) -> int:
            return (framework.n_params(model) * self.__width) // 8
        return f
