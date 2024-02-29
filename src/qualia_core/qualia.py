from __future__ import annotations

import inspect
import sys
from dataclasses import dataclass
from typing import Any, TypeVar

import colorful as cf  # type: ignore[import-untyped]

from qualia_core.learningmodel.LearningModel import LearningModel
from qualia_core.typing import TYPE_CHECKING, ModelParamsConfigDict, OptimizerConfigDict

if TYPE_CHECKING:
    from qualia_core.dataaugmentation.DataAugmentation import DataAugmentation  # noqa: TCH001
    from qualia_core.datamodel.DataModel import DataModel  # noqa: TCH001
    from qualia_core.datamodel.RawDataModel import RawData, RawDataModel  # noqa: TCH001
    from qualia_core.experimenttracking.ExperimentTracking import ExperimentTracking  # noqa: TCH001
    from qualia_core.learningframework.LearningFramework import LearningFramework  # noqa: TCH001
    from qualia_core.typing import RecursiveConfigDict

@dataclass
class TrainResult:
    name: str
    i: int
    model: Any
    params: int
    mem_params: int
    acc: float
    metrics: dict[str, Any]
    datamodel: DataModel[RawData]
    trainset: RawData
    testset: RawData
    framework: LearningFramework[Any]
    batch_size: int
    optimizer: Any
    log: bool
    dataaugmentations: list[DataAugmentation]
    experimenttracking: ExperimentTracking | None

T = TypeVar('T', bound=LearningModel)

def gen_tag(mname: str, q: str, o: int, i: int, c: int) -> str:
    return f'{mname}_q{q}_o{o}_c{c}_r{i}'

def instantiate_model(dataset: RawData,  # noqa: PLR0913
                      framework: LearningFramework[T],
                      model: type[T],
                      model_params: ModelParamsConfigDict | None,
                      model_name: str,
                      iteration: int,
                      load: bool = True) -> T:  # noqa: FBT001, FBT002
    model_params = model_params if model_params is not None else ModelParamsConfigDict()

    if 'input_shape' not in model_params:
        model_params['input_shape'] = dataset.x.shape[1:]
    else:
        model_params['input_shape'] = tuple(model_params['input_shape'])
    if 'output_shape' not in model_params:
        model_params['output_shape'] = dataset.y.shape[1:]
    else:
        model_params['output_shape'] = tuple(model_params['output_shape'])

    if 'iteration' in inspect.signature(model).parameters:
        model_params['iteration'] = iteration

    # Instantiate model
    new_model = model(**model_params)

    if load:
        new_model = framework.load(f'{model_name}_r{iteration}', new_model)

    print(f'{new_model.input_shape=} {new_model.output_shape=}')

    # Show model architecture
    framework.summary(new_model)

    return new_model

def train(datamodel: RawDataModel,  # noqa: PLR0913
          train_epochs: int,
          iteration: int,
          framework: LearningFramework[T],
          model: type[T],
          model_name: str,
          model_params: RecursiveConfigDict | None = None,
          batch_size: int | None = None,
          optimizer: OptimizerConfigDict | None = None,
          load: bool = False,  # noqa: FBT002, FBT001
          train: bool = True,  # noqa: FBT001, FBT002
          evaluate: bool = True,  # noqa: FBT001, FBT002
          dataaugmentations: list[DataAugmentation] | None = None,
          experimenttracking: ExperimentTracking | None = None,
          use_test_as_valid: bool = False) -> TrainResult:  # noqa: FBT001, FBT002

    if batch_size is None:
        batch_size = 32

    new_model = instantiate_model(dataset=datamodel.sets.train,
                       framework=framework,
                       model=model,
                       model_params=model_params,
                       model_name=model_name,
                       iteration=iteration,
                       load=load)

    # Export model visualization to dot file
    framework.save_graph_plot(new_model, f'{model_name}_r{iteration}')

    # You can plot the quantize training graph on tensorboard
    if train:
        new_model = framework.train(new_model,
                        trainset=datamodel.sets.train,
                        validationset=datamodel.sets.valid if not use_test_as_valid else datamodel.sets.test,
                        epochs=train_epochs,
                        batch_size=batch_size,
                        optimizer=optimizer,
                        dataaugmentations=dataaugmentations,
                        experimenttracking=experimenttracking,
                        name=f'{model_name}_r{iteration}_train')

    metrics = {}
    if evaluate:
        print(f'{cf.bold}Evaluation on train dataset{cf.reset}')
        metrics = framework.evaluate(new_model,
                                 datamodel.sets.train,
                                 batch_size=batch_size,
                                 dataaugmentations=dataaugmentations,
                                 experimenttracking=experimenttracking,
                                 dataset_type='train',
                                 name=f'{model_name}_r{iteration}_eval_train')

        if len(datamodel.sets.test.x) > 0: # Don't evaluate if testset is empty
            print(f'{cf.bold}Evaluation on test dataset{cf.reset}')
            metrics = framework.evaluate(new_model,
                                     datamodel.sets.test,
                                     batch_size=batch_size,
                                     dataaugmentations=dataaugmentations,
                                     experimenttracking=experimenttracking,
                                     dataset_type='test',
                                     name=f'{model_name}_r{iteration}_eval_test')

    # Do not save loaded model that hasn't been retrained
    if train or not load:
        framework.export(new_model, f'{model_name}_r{iteration}')

    return TrainResult(name=model_name,
                       i=iteration,
                       model=new_model,
                       params=framework.n_params(new_model),
                       mem_params=framework.n_params(new_model) * 4, # Non-quantized model is assumed to be 32 bits
                       acc=metrics.get('testacc', None),
                       metrics=metrics,
                       datamodel=datamodel,
                       trainset=datamodel.sets.train,
                       testset=datamodel.sets.test,
                       framework=framework,
                       batch_size=batch_size,
                       optimizer=optimizer,
                       log=True,
                       dataaugmentations=dataaugmentations,
                       experimenttracking=experimenttracking)

def prepare_deploy(
    datamodel,
    model_kind,
    model_name,
    model,
    framework,
    iteration,
    deploy_target,
    quantize='float32',
    optimize=None,
    compress=1,
    tag='main',
    converter=None,
    converter_params={},
    deployers=None,
    deployer_params={},
    representative_dataset=None):

    if not converter: # no custom converter passed as parameter, check if model suggests custom converter
        converter = getattr(model_kind, 'converter', False)

    if converter:
        ca = converter(quantize=quantize, **converter_params).convert(framework, model, f'{model_name}_r{iteration}', representative_dataset=representative_dataset)
    else: # No conversion taking place since no converter was specified
        ca = model

    if ca is None:
        return None

    if not deployers:
        if not converter:
            print('Error: no converter and no deployers specified', file=sys.stderr)
            return None
        elif not hasattr(ca, 'deployers'):
            print('Error: no deployers specified and converter does not suggest a deployer', file=sys.stderr)
            return None
        else:
            deployers = ca.deployers
    return getattr(deployers, deploy_target)(**deployer_params).prepare(tag=tag, model=ca, optimize=optimize, compression=compress)

def deploy(model_kind, deploy_target, tag='main', deployers=None, deployer_params={}):
    if not deployers: # no custom deployers passed as parameter, check if model suggests custom converter
        converter = getattr(model_kind, 'converter', False)
        if converter and converter.deployers: # Converter suggested deployers
            deployers = converter.deployers

    return getattr(deployers, deploy_target)(**deployer_params).deploy(tag=tag)

def evaluate(
    datamodel,
    model_kind,
    model_name,
    model,
    framework,
    iteration,
    target,
    quantization,
    fmem_params,
    tag,
    limit=None,
    evaluator=None,
    evaluator_params={},
    dataaugmentations=None):

    if not evaluator: # no custom deployers passed as parameter, check if model suggests custom converter
        converter = getattr(model_kind, 'converter', False)
        if converter and converter.evaluator: # Converter suggested deployers
            evaluator = converter.evaluator

    if not evaluator:
        raise ValueError('No evaluator')
    result = evaluator(**evaluator_params).evaluate(framework=framework,
                                                    model_kind=model_kind,
                                                    dataset=datamodel,
                                                    target=target,
                                                    tag=tag,
                                                    limit=limit,
                                                    dataaugmentations=[da for da in dataaugmentations if da.evaluate])
    if not result:
        return result

    # fill in iteration, name quantization from context
    result.name = model_name
    result.i = iteration
    result.quantization = quantization
    # fill in params count and memory from model
    result.params = framework.n_params(model)
    result.mem_params = fmem_params(framework, model)

    return result
