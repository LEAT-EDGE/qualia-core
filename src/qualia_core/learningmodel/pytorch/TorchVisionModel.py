import torch
import torch.nn as nn
from torch.fx import Tracer, GraphModule

from qualia_core.learningmodel.pytorch.layers import layers as custom_layers


class TorchVisionModel(nn.Module):
    # Custom tracer that generates call_module for our custom Qualia layers instead of attempting to trace their forward()
    class TracerCustomLayers(Tracer):
        def __init__(self, custom_layers: tuple, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.custom_layers = custom_layers

        def is_leaf_module(self, m: torch.nn.Module, module_qualified_name : str) -> bool:
            return super().is_leaf_module(m, module_qualified_name) or isinstance(m, custom_layers)

    def _shape_channels_last_to_first(self, shape):
        return (shape[-1], ) + shape[0:-1]

    def __init__(self, input_shape, output_shape, model, *args, **kwargs):
        import torchvision.models as models

        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.pretrained_model = getattr(models, model)(*args, **kwargs)

        self.fm = self.create_feature_extractor(self.pretrained_model, 'avgpool')
        for param in self.fm.parameters():
            param.requires_grad = False

        self.fm_shape = self.fm(torch.rand((1, *self._shape_channels_last_to_first(input_shape)))).shape

        print(tuple(self.fm_shape[1:]),self.output_shape)

        self.linear = nn.Linear(self.fm_shape[1], self.output_shape[0])
        self.flatten = nn.Flatten()

        for name, param in self.fm.named_parameters():
            print(name, param.requires_grad)

    # Similar to torchvision's but simplified for our specific use case
    def create_feature_extractor(self, model: nn.Module, return_node: str):
        # Feature extractor only used in eval mode
        model.eval()

        tracer = self.TracerCustomLayers(custom_layers=custom_layers)
        graph = tracer.trace(model)
        graph.print_tabular()
        graphmodule = GraphModule(tracer.root, graph, tracer.root.__class__.__name__)

        # Remove existing output node
        old_output = [n for n in graphmodule.graph.nodes if n.op == 'output']
        if not old_output:
            raise ValueError(f'No output in dl model')
        if len(old_output) > 1:
            raise ValueError(f'Multiple outputs in dl model')
        graphmodule.graph.erase_node(old_output[0])
        print(f'{old_output=}')

        # Find desired output layer
        new_output = [n for n in graphmodule.graph.nodes if n.name == return_node]
        if not new_output:
            raise ValueError(f'fm_output = \'{return_node}\' not found in dl model')
        if len(new_output) > 1:
            raise ValueError(f'Multiple matches for fm_output = \'{return_node}\' in dl model')

        # Add new output for desired layer
        with graphmodule.graph.inserting_after(list(graphmodule.graph.nodes)[-1]):
            graphmodule.graph.output(new_output[0])

        # Remove unused layers
        graphmodule.graph.eliminate_dead_code()

        graphmodule.recompile()

        graphmodule.graph.print_tabular()

        return graphmodule

    def forward(self, x):
        x = self.fm(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x
