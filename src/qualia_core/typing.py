from __future__ import annotations

import os
import sys
import typing

if sys.version_info >= (3, 12):
    from typing import NotRequired, TypeAliasType, TypedDict
else:
    from typing_extensions import NotRequired, TypeAliasType, TypedDict

TYPE_CHECKING = typing.TYPE_CHECKING or os.environ.get('SPHINX_AUTODOC', False)



if TYPE_CHECKING:
    RecursiveConfigUnion = typing.Union[dict[str, 'RecursiveConfigUnion'],
                                        list['RecursiveConfigUnion'],
                                        str,
                                        int,
                                        float,
                                        bool]
else:
    RecursiveConfigUnion = TypeAliasType('RecursiveConfigUnion', typing.Union[dict[str, 'RecursiveConfigUnion'],
                                        list['RecursiveConfigUnion'],
                                        str,
                                        int,
                                        float,
                                        bool])
RecursiveConfigDict = dict[str, 'RecursiveConfigUnion']


class GenericModuleConfigDict(TypedDict):
    kind: str
    params: NotRequired[RecursiveConfigDict]

class BenchConfigDict(TypedDict):
    name: str
    seed: int
    first_run: int
    last_run: int
    plugins: NotRequired[list[str]]
    use_test_as_valid: NotRequired[bool]

class ExperimentTrackingConfigDict(GenericModuleConfigDict):
    ...

class LearningFrameworkConfigDict(GenericModuleConfigDict):
    ...

class ConverterConfigDict(GenericModuleConfigDict):
    ...

class DeployerConfigDict(TypedDict):
    kind: NotRequired[str] # Not mandatory since deployer can be suggested by converter.
    params: NotRequired[RecursiveConfigDict]

class EvaluatorConfigDict(TypedDict):
    kind: NotRequired[str] # Not mandatory since evaluator can be suggested by deployer
    params: NotRequired[RecursiveConfigDict]

class DeployConfigDict(TypedDict):
    target: str
    converter: ConverterConfigDict
    deployer: NotRequired[DeployerConfigDict]
    evaluator: NotRequired[EvaluatorConfigDict]
    quantize: list[str]
    optimize: NotRequired[list[str]]
    compress: NotRequired[list[int]]
    limit: NotRequired[int]

class DatasetConfigDict(GenericModuleConfigDict):
    ...

class PreprocessingConfigDict(GenericModuleConfigDict):
    ...

class DataAugmentationConfigDict(GenericModuleConfigDict):
    ...

class PostprocessingConfigDict(GenericModuleConfigDict):
    export: NotRequired[bool]

class SchedulerConfigDict(GenericModuleConfigDict):
    ...

class OptimizerConfigDict(GenericModuleConfigDict):
    scheduler: NotRequired[SchedulerConfigDict]

class QuantizerConfigDict(TypedDict):
    quant_enable: NotRequired[bool]
    LSQ: NotRequired[bool]
    bits: NotRequired[int]
    force_q: NotRequired[int]
    range_setting: NotRequired[str]
    roundtype: NotRequired[str]
    tensor_type: NotRequired[str]
    quantype: NotRequired[str]
    is_asymmetric: NotRequired[bool]

class QuantizationConfigDict(QuantizerConfigDict):
    act: NotRequired[QuantizerConfigDict]
    v: NotRequired[QuantizerConfigDict]
    input: NotRequired[QuantizerConfigDict]  # noqa: A003
    w: NotRequired[QuantizerConfigDict]
    bias: NotRequired[QuantizerConfigDict]

class ModelParamsConfigDict(TypedDict):
    input_shape: typing.Union[list[int], tuple[int, ...]]
    output_shape: typing.Union[list[int], tuple[int, ...]]
    quant_params: NotRequired[QuantizationConfigDict]

class UpdateConfigDict(TypedDict):
    quant_params: NotRequired[QuantizationConfigDict]

class ModelCommonConfigDict(TypedDict):
    params: NotRequired[typing.Union[RecursiveConfigDict, ModelParamsConfigDict]]
    epochs: NotRequired[int]
    batch_size: NotRequired[int]
    load: NotRequired[bool]
    train: NotRequired[bool]
    evaluate: NotRequired[bool]
    optimizer: NotRequired[OptimizerConfigDict]
    update: NotRequired[UpdateConfigDict]

class ModelTemplateConfigDict(ModelCommonConfigDict):
    kind: NotRequired[str]

class ModelConfigDict(ModelCommonConfigDict):
    kind: str
    name: str
    disabled: NotRequired[bool]

class ParameterResearchStudyConfigDict(TypedDict):
    load: NotRequired[bool]
    params: RecursiveConfigDict

class ParameterResearchOptimizeConfigDict(TypedDict):
    params: RecursiveConfigDict

class ParameterResearchConfigDict(TypedDict):
    optimize: ParameterResearchOptimizeConfigDict
    study: ParameterResearchStudyConfigDict
    trial: RecursiveConfigDict

class ConfigDict(TypedDict):
    bench: BenchConfigDict
    experimenttracking: NotRequired[ExperimentTrackingConfigDict]
    learningframework: LearningFrameworkConfigDict
    deploy: NotRequired[DeployConfigDict]
    dataset: DatasetConfigDict

    preprocessing: NotRequired[list[PreprocessingConfigDict]]
    data_augmentation: NotRequired[list[GenericModuleConfigDict]]
    postprocessing: NotRequired[list[PostprocessingConfigDict]]
    model: list[ModelConfigDict]
    model_template: ModelTemplateConfigDict
    parameter_research: NotRequired[ParameterResearchConfigDict]

__all__ = ['TYPE_CHECKING', 'RecursiveConfigDict']
