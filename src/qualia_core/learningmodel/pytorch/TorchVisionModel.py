from __future__ import annotations

import logging
import sys
from typing import Any

import torch
from torch import nn
from torch.fx import GraphModule, Tracer

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
            return super().is_leaf_module(m, module_qualified_name) or isinstance(m, custom_layers)

    def _shape_channels_last_to_first(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        return (shape[-1], ) + shape[0:-1]

    def __init__(self,  # noqa: PLR0913
                 input_shape: tuple[int, ...],
                 output_shape: tuple[int, ...],
                 model: str,
                 fm_output_layer: str = 'avgpool',
                 freeze_feature_extractor: bool = True,  # noqa: FBT001, FBT002
                 *args: Any, **kwargs: Any) -> None:  # noqa: ANN401 We need to pass whatever arg to TorchVision
        from torchvision import models  # type: ignore[import-untyped]

        super().__init__(input_shape=input_shape, output_shape=output_shape)

        self.pretrained_model = getattr(models, model)(*args, **kwargs)

        self.fm = self.create_feature_extractor(self.pretrained_model, fm_output_layer)
        for param in self.fm.parameters():
            param.requires_grad = not freeze_feature_extractor

        self.fm_shape = self.fm(torch.rand((1, *self._shape_channels_last_to_first(input_shape)))).shape

        self.linear = nn.Linear(self.fm_shape[1], self.output_shape[0])
        self.flatten = nn.Flatten()

        for name, param in self.fm.named_parameters():
            logger.debug('Layer: %s, trainable: %s.', name, param.requires_grad)

    # Similar to torchvision's but simplified for our specific use case
    def create_feature_extractor(self, model: nn.Module, return_node: str) -> GraphModule:
        # Feature extractor only used in eval mode
        _ = model.eval()

        tracer = self.TracerCustomLayers(custom_layers=custom_layers)
        graph = tracer.trace(model)
        graph.print_tabular()
        graphmodule = GraphModule(tracer.root, graph, tracer.root.__class__.__name__)

        # Remove existing output node
        old_output = [n for n in graphmodule.graph.nodes if n.op == 'output']
        if not old_output:
            logger.error('No output found in TorchVision model.')
            raise RuntimeError
        if len(old_output) > 1:
            logger.error('Multiple outputs found in TorchVision model.')
            raise RuntimeError
        logger.info("Removing output '%s'", old_output)
        graphmodule.graph.erase_node(old_output[0])

        # Find desired output layer
        new_output = [n for n in graphmodule.graph.nodes if n.name == return_node]
        if not new_output:
            logger.error("fm_output_layer '%s' not found in TorchVision model.", return_node)
            raise ValueError
        if len(new_output) > 1:
            logger.error("Multiple matches for fm_output_layer '%s'", return_node)
            raise RuntimeError

        # Add new output for desired layer
        with graphmodule.graph.inserting_after(list(graphmodule.graph.nodes)[-1]):
            _ = graphmodule.graph.output(new_output[0])

        # Remove unused layers
        _ = graphmodule.graph.eliminate_dead_code()

        _ = graphmodule.recompile()

        graphmodule.graph.print_tabular()

        return graphmodule

    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        x = self.fm(input)
        x = self.flatten(x)
        return self.linear(x)
