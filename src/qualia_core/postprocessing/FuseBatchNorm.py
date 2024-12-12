from __future__ import annotations

import copy
import dataclasses
import logging
import sys

from torch import nn
from torch.fx.graph_module import GraphModule

from qualia_core.learningframework.PyTorch import PyTorch
from qualia_core.typing import TYPE_CHECKING

from .PostProcessing import PostProcessing

if TYPE_CHECKING:
    from torch.fx.graph import Graph  # noqa: TC002

    from qualia_core.learningframework.LearningFramework import LearningFramework  # noqa: TC001
    from qualia_core.qualia import TrainResult  # noqa: TC001
    from qualia_core.typing import ModelConfigDict

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)


class FuseBatchNorm(PostProcessing[nn.Module]):
    extra_custom_layers: tuple[type[nn.Module], ...] = ()

    patterns: list[tuple[type[nn.Module], type[nn.Module]]]

    def __init__(self, evaluate: bool = True) -> None:  # noqa: FBT001, FBT002
        super().__init__()
        self.__evaluate = evaluate

        from torch import nn

        self.patterns = [
                (nn.Conv1d, nn.BatchNorm1d),
                (nn.Conv2d, nn.BatchNorm2d),
                (nn.Conv3d, nn.BatchNorm3d),
                ]

    # Copied from torch/fx/experimental/optimization.py because we need a custom Tracer for our custom layers
    def fuse(self, model: nn.Module,
             graphmodule_cls: type[GraphModule],
             framework: PyTorch,
             inplace: bool = False) -> GraphModule:  # noqa: FBT001, FBT002
        """Fuses convolution/BN layers for inference purposes.

        Will deepcopy your model by default, but can modify the model inplace as well.
        """
        from torch.fx.experimental.optimization import matches_module_pattern, replace_node_module
        from torch.nn.utils.fusion import fuse_conv_bn_eval

        if not inplace:
            model = copy.deepcopy(model)

        graph, graphmodule = framework.trace_model(model, extra_custom_layers=self.extra_custom_layers)
        graph.print_tabular()
        fx_model = graphmodule

        modules: dict[str, nn.Module] = dict(fx_model.named_modules())
        new_graph: Graph = copy.deepcopy(fx_model.graph)

        for pattern in self.patterns:
            for node in new_graph.nodes:
                if matches_module_pattern(pattern, node, modules):
                    if len(node.args[0].users) > 1:  # Output of conv is used by other nodes
                        continue
                    conv = modules[node.args[0].target]
                    bn = modules[node.target]
                    if not bn.track_running_stats:
                        continue
                    fused_conv = fuse_conv_bn_eval(conv, bn)
                    replace_node_module(node.args[0], modules, fused_conv)
                    node.replace_all_uses_with(node.args[0])
                    new_graph.erase_node(node)
        return graphmodule_cls(fx_model, new_graph)

    @override
    def __call__(self, trainresult: TrainResult, model_conf: ModelConfigDict) -> tuple[TrainResult, ModelConfigDict]:
        if not isinstance(trainresult.framework, PyTorch):
            logger.error('Only available for PyTorch-based LearningFramework')
            raise TypeError

        model = trainresult.model
        model.eval() # Can only fuse models in eval mode

        fused_model = self.fuse(model,
                                graphmodule_cls=GraphModule,
                                framework=trainresult.framework,
                                inplace=True)

        # Copy input_shape/output_shape into generated model
        fused_model.input_shape = model.input_shape
        fused_model.output_shape = model.output_shape

        model_name = f'{trainresult.name}_fused'

        logger.info('Model after BatchNorm/Conv fusion')
        trainresult.framework.summary(fused_model)

        acc = trainresult.acc
        if self.__evaluate:
            logger.info('Evaluation on test dataset after BatchNorm/Conv fusion')
            metrics = trainresult.framework.evaluate(fused_model,
                                                     trainresult.testset,
                                                     batch_size=trainresult.batch_size,
                                                     dataaugmentations=trainresult.dataaugmentations,
                                                     experimenttracking=trainresult.experimenttracking,
                                                     dataset_type='test',
                                                     name=model_name)
            acc = metrics.get('testacc', None)

        new_model_conf = copy.deepcopy(model_conf)
        new_model_conf.get('params', {})['batch_norm'] = False

        return (dataclasses.replace(trainresult, name=model_name, model=fused_model, acc=acc),
                new_model_conf)

    @override
    def process_model(self,
                      model: nn.Module,
                      model_conf: ModelConfigDict,
                      framework: LearningFramework[nn.Module]) -> tuple[nn.Module, ModelConfigDict]:
        if not isinstance(framework, PyTorch):
            logger.error('Only available for PyTorch-based LearningFramework')
            raise TypeError

        model = model.eval() # Can only fuse models in eval mode
        fused_model = self.fuse(model, graphmodule_cls=GraphModule, framework=framework, inplace=True)
        # Copy input_shape/output_shape into generated model
        fused_model.input_shape = model.input_shape
        fused_model.output_shape = model.output_shape
        model_conf.get('params', {})['batch_norm'] = False
        return fused_model, model_conf

    @override
    def process_name(self, name: str) -> str:
        return f'{name}_fused'
