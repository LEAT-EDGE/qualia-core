from __future__ import annotations

import copy
import logging
import operator
import sys
from typing import Any, Callable, ClassVar

import torch
from torch import nn
from torch.fx import Graph, GraphModule, Node, Tracer
from torch.fx.experimental.optimization import replace_node_module

from qualia_core.learningmodel.pytorch import layers as qualia_layers
from qualia_core.learningmodel.pytorch.layers import layers as custom_layers
from qualia_core.learningmodel.pytorch.layers import quantized_layers
from qualia_core.learningmodel.pytorch.layers.quantized_layers import quantized_layers as quantized_layers_list
from qualia_core.typing import TYPE_CHECKING, ModelConfigDict, ModelParamsConfigDict, QuantizationConfigDict
from qualia_core.utils.merge_dict import merge_dict

from .QuantizationAwareTraining import QuantizationAwareTraining

if TYPE_CHECKING:
    from qualia_core.learningframework.LearningFramework import LearningFramework  # noqa: TCH001
    from qualia_core.learningmodel.pytorch.layers.QuantizedLayer import QuantizedLayer  # noqa: TCH001

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)

class QuantizationAwareTrainingFX(QuantizationAwareTraining):
    FUSE_ACTIVATIONS: ClassVar[tuple[type[nn.Module], ...]] = (torch.nn.ReLU, torch.nn.ReLU6)
    QUANTIZED_LAYERS: ClassVar[dict[type[nn.Module], type[QuantizedLayer]]] = {
            nn.AdaptiveAvgPool1d: quantized_layers.QuantizedAdaptiveAvgPool1d,
            nn.AdaptiveAvgPool2d: quantized_layers.QuantizedAdaptiveAvgPool2d,
            nn.AvgPool1d: quantized_layers.QuantizedAvgPool1d,
            nn.AvgPool2d: quantized_layers.QuantizedAvgPool2d,
            nn.BatchNorm1d: quantized_layers.QuantizedBatchNorm1d,
            nn.BatchNorm2d: quantized_layers.QuantizedBatchNorm2d,
            nn.Conv1d: quantized_layers.QuantizedConv1d,
            nn.Conv2d: quantized_layers.QuantizedConv2d,
            nn.Linear: quantized_layers.QuantizedLinear,
            nn.MaxPool1d: quantized_layers.QuantizedMaxPool1d,
            nn.MaxPool2d: quantized_layers.QuantizedMaxPool2d,
            nn.ReLU: quantized_layers.QuantizedReLU,
            nn.ReLU6: quantized_layers.QuantizedReLU,
            qualia_layers.Add: quantized_layers.QuantizedAdd,
            qualia_layers.CustomBatchNorm1d: quantized_layers.QuantizedBatchNorm1d,
            qualia_layers.CustomBatchNorm2d: quantized_layers.QuantizedBatchNorm2d,
            qualia_layers.GlobalSumPool1d: quantized_layers.QuantizedGlobalSumPool1d,
            qualia_layers.GlobalSumPool2d: quantized_layers.QuantizedGlobalSumPool2d,
    }
    QUANTIZED_FUNCTIONS: ClassVar[dict[Callable[..., Any], Callable[[tuple[Any, ...], QuantizationConfigDict], nn.Module]]] = {
        operator.add: lambda _, quant_params: quantized_layers.QuantizedAdd(quant_params=quant_params),
        torch.nn.functional.adaptive_avg_pool2d: lambda args, quant_params: quantized_layers.QuantizedAdaptiveAvgPool2d(args[1],
                                                                                                                        quant_params=quant_params),
    }
    FUNCTION_INPUT_ARG_INDEX: ClassVar[dict[Callable[..., Any] | str, tuple[int, ...]]] = {
        operator.add: (0, 1),
        torch.nn.functional.adaptive_avg_pool2d: (0,),
    }

    class TracerCustomLayers(Tracer):
        def __init__(self, custom_layers: tuple[type[nn.Module], ...]) -> None:
            super().__init__()
            self.custom_layers = custom_layers

        @override
        def is_leaf_module(self, m: torch.nn.Module, module_qualified_name : str) -> bool:
            return super().is_leaf_module(m, module_qualified_name) or isinstance(m, self.custom_layers)

    def _get_modules(self, model: nn.Module) -> dict[str, nn.Module]:
        return dict(model.named_modules())

    def _fuse_activation(self, graph: Graph,
                         model: nn.Module,
                         node: Node) -> None:
        if not isinstance(node.target, str):
            logger.warning('Could not fuse activation %s with previous layer, expected target to be str, got %s',
                           node.name,
                           type(node.target))
            return

        module = self._get_modules(model)[node.target]

        previous_node = node.args[0]

        if not isinstance(previous_node, Node):
            logger.warning('Could not fuse activation %s with previous layer, expected first argument to be Node, got %s',
                           node.name,
                           type(previous_node))
            return

        if not isinstance(previous_node.target, str):
            logger.warning('Could not fuse activation %s with previous layer, expected previous node target to be str, got %s',
                           node.name,
                           type(node.target))
            return
        # Fuse ReLU
        previous_module = self._get_modules(model)[previous_node.target]
        # Should support fused activation if activation attribute is declared
        if hasattr(previous_module, 'activation'):
            previous_module.activation = module
            _ = node.replace_all_uses_with(previous_node)
            graph.erase_node(node)

    def _replace_module_with_quantized_layer(self,
                                             model: nn.Module,
                                             node: Node,
                                             quant_params: QuantizationConfigDict) -> None:
        if not isinstance(node.target, str):
            logger.error('Could replace module %s with quantized layer, expected target to be str, got %s',
                           node.name,
                           type(node.target))
            raise TypeError
        # Instantiate a Quantized module from a "regular" module attributes, from_module() should also copy weights if applicable
        module = self._get_modules(model)[node.target]
        quantized_module = self.QUANTIZED_LAYERS[type(module)].from_module(module, quant_params)
        if not isinstance(quantized_module, nn.Module):
            logger.error('Quantized layer must be a torch.nn.Module')
            raise TypeError
        replace_node_module(node, self._get_modules(model), quantized_module)

    def _replace_function_with_quantized_layer(self,
                                               graph: Graph,
                                               model: nn.Module,
                                               node: Node,
                                               quant_params: QuantizationConfigDict) -> None:
        if not callable(node.target):
            logger.error('Could replace module %s with quantized layer, expected target to be callable, got %s',
                           node.name,
                           type(node.target))
            raise TypeError
        setattr(model, node.name, self.QUANTIZED_FUNCTIONS[node.target](node.args, quant_params=quant_params))
        with graph.inserting_after(node):
            args = tuple(node.args[i] for i in self.FUNCTION_INPUT_ARG_INDEX[node.target])
            new_node = graph.call_module(node.name, args, node.kwargs)
            _ = node.replace_all_uses_with(new_node)
        # Remove the old node from the graph
        graph.erase_node(node)


    def _build_quantized_model_fx(self, model: nn.Module, quant_params: QuantizationConfigDict) -> nn.Module:
        tracer = self.TracerCustomLayers(custom_layers=custom_layers)
        graph = tracer.trace(model)
        graph.print_tabular()

        for node in graph.nodes:
            if node.op == 'call_module':
                module = self._get_modules(model)[node.target]
                if isinstance(module, quantized_layers_list):
                    logger.info('Node %s with module type %s already quantized, skipping', node.name, type(module))
                elif isinstance(module, self.FUSE_ACTIVATIONS): # Priorize activation fusion over quantization
                    self._fuse_activation(graph, model, node)
                elif type(module) in self.QUANTIZED_LAYERS:
                    self._replace_module_with_quantized_layer(model, node, quant_params)
                else:
                    logger.warning('Node %s not quantized, unknown module type %s', node.name, type(module))
            elif node.op == 'call_function':
                if node.target in self.QUANTIZED_FUNCTIONS:
                    self._replace_function_with_quantized_layer(graph, model, node, quant_params)
                else:
                    logger.warning('Node %s not quantized, unknown function %s', node.name, node.target)


        graphmodule = GraphModule(tracer.root, graph, tracer.root.__class__.__name__)

        graphmodule.graph.print_tabular()

        return graphmodule

    @override
    def _build_quantized_model(self,
                               model: nn.Module,
                               framework: LearningFramework[Any],
                               model_conf: ModelConfigDict) -> tuple[nn.Module, ModelParamsConfigDict]:
        logger.info('Building quantized model')
        # Create quantized model with parameters from original model
        # Copy params so that original config is not lost for subsequent calls with different models

        qmodel_params: ModelParamsConfigDict = copy.deepcopy(self.qmodel_params)

        if 'update' in model_conf :
            qmodel_params['quant_params'].update(model_conf['update']['quant_params'])

        if 'params' in model_conf:
            qmodel_params = merge_dict(qmodel_params, model_conf['params'])

        self.lsq: bool = qmodel_params['quant_params'].get('LSQ', False)
        self.quant_framework: str | None = qmodel_params['quant_params'].get('quant_framework', None)
        self.width: int = qmodel_params['quant_params'].get('bits', 0)
        self.force_q: int | None = qmodel_params['quant_params'].get('force_q', None)

        quantized_model = self._build_quantized_model_fx(model, quant_params=qmodel_params['quant_params'])

        framework.summary(quantized_model)

        return quantized_model, qmodel_params
