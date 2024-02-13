from __future__ import annotations

import logging
import sys
from typing import Any

import torch
from torch import nn
from torch.fx import Graph, GraphModule, Node, Tracer

from qualia_core.learningmodel.pytorch.layers import layers as custom_layers
from qualia_core.learningmodel.pytorch.LearningModelPyTorch import LearningModelPyTorch

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)

class TorchVisionModel(LearningModelPyTorch):
    # Custom tracer that generates call_module for our custom Qualia layers instead of attempting to trace their forward()
    class TracerCustomLayers(Tracer):
        def __init__(self, custom_layers: tuple[type[nn.Module], ...]) -> None:
            super().__init__()
            self.custom_layers = custom_layers

        @override
        def is_leaf_module(self, m: torch.nn.Module, module_qualified_name : str) -> bool:
            return super().is_leaf_module(m, module_qualified_name) or isinstance(m, self.custom_layers)

    def _shape_channels_last_to_first(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        return (shape[-1], ) + shape[0:-1]

    def __init__(self,  # noqa: PLR0913
                 input_shape: tuple[int, ...],
                 output_shape: tuple[int, ...],
                 model: str,
                 replace_classifier: bool = True,  # noqa: FBT002, FBT001
                 fm_output_layer: str = 'flatten',
                 freeze_feature_extractor: bool = True,  # noqa: FBT001, FBT002
                 *args: Any, **kwargs: Any) -> None:  # noqa: ANN401 We need to pass whatever arg to TorchVision
        from torchvision import models  # type: ignore[import-untyped]

        super().__init__(input_shape=input_shape, output_shape=output_shape)

        self.replace_classifier = replace_classifier
        pretrained_model = getattr(models, model)(*args, **kwargs)

        if replace_classifier:
            self.fm = self.create_feature_extractor(pretrained_model, fm_output_layer)
            for param in self.fm.parameters():
                param.requires_grad = not freeze_feature_extractor

            self.fm_shape = self.fm(torch.rand((1, *self._shape_channels_last_to_first(input_shape)))).shape

            self.linear = nn.Linear(self.fm_shape[1], self.output_shape[0])
        else:
            self.model = pretrained_model

        for name, param in self.named_parameters():
            logger.info('Layer: %s, trainable: %s.', name, param.requires_grad)

    def recursive_erase_node(self, graph: Graph, nodes: list[Node]) -> None:
        for node in nodes:
            self.recursive_erase_node(graph, list(node.users.keys()))
            logger.info('Removing %s', node)
            graph.erase_node(node)

    # Similar to torchvision's but simplified for our specific use case
    def create_feature_extractor(self, model: nn.Module, return_node: str) -> GraphModule:
        # Feature extractor only used in eval mode
        _ = model.eval()

        tracer = self.TracerCustomLayers(custom_layers=custom_layers)
        graph = tracer.trace(model)
        graph.print_tabular()

        # Find desired output layer
        new_output = [n for n in graph.nodes if n.name == return_node]
        if not new_output:
            logger.error("fm_output_layer '%s' not found in TorchVision model.", return_node)
            raise ValueError
        if len(new_output) > 1:
            logger.error("Multiple matches for fm_output_layer '%s'", return_node)
            raise RuntimeError

        logger.info('Removing all nodes after %s', new_output[0])
        self.recursive_erase_node(graph, list(new_output[0].users.keys()))

        # Add new output for desired layer
        with graph.inserting_after(list(graph.nodes)[-1]):
            logger.info('Setting %s as output of the feature extractor', new_output[0])
            _ = graph.output(new_output[0])

        graphmodule = GraphModule(tracer.root, graph, tracer.root.__class__.__name__)

        graphmodule.graph.print_tabular()

        return graphmodule

    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        if self.replace_classifier:
            x = self.fm(input)
            return self.linear(x)

        return self.model(input)
