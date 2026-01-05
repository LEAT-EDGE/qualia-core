from __future__ import annotations

import inspect
import sys
from dataclasses import dataclass
from typing import Any, TypeVar

import colorful as cf  # type: ignore[import-untyped]

from qualia_core.learningmodel.LearningModel import LearningModel
from qualia_core.typing import TYPE_CHECKING, DeployerConfigDict, ModelParamsConfigDict, OptimizerConfigDict

if TYPE_CHECKING:
    from pathlib import Path
    from types import ModuleType

    from qualia_core.dataaugmentation.DataAugmentation import DataAugmentation
    from qualia_core.datamodel.DataModel import DataModel
    from qualia_core.datamodel.RawDataModel import RawData, RawDataModel
    from qualia_core.deployment.Deploy import Deploy
    from qualia_core.deployment.Deployer import Deployer
    from qualia_core.experimenttracking.ExperimentTracking import ExperimentTracking
    from qualia_core.learningframework.LearningFramework import LearningFramework
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
    parent_model_hash: str | None
    model_hash: str | None


T = TypeVar('T', bound=LearningModel)


def gen_tag(mname: str, q: str, o: int, i: int, c: int) -> str:
    return f'{mname}_q{q}_o{o}_c{c}_r{i}'


def instantiate_model(dataset: RawData,  # noqa: PLR0913
                      framework: LearningFramework[T],
                      model: type[T],
                      model_params: ModelParamsConfigDict | None,
                      model_name: str,
                      iteration: int,
                      load: bool = True) -> tuple[T, Path | None]:  # noqa: FBT001, FBT002
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

    print(f'{new_model.input_shape=} {new_model.output_shape=}')

    # Show model architecture
    framework.summary(new_model)

    loaded_path: Path | None = None
    if load:
        new_model, loaded_path = framework.load(f'{model_name}_r{iteration}', new_model)

    return new_model, loaded_path

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
    model_path: Path | None = None

    if batch_size is None:
        batch_size = 32

    new_model, model_path = instantiate_model(dataset=datamodel.sets.train,
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
            test_metrics = framework.evaluate(new_model,
                                         datamodel.sets.test,
                                         batch_size=batch_size,
                                         dataaugmentations=dataaugmentations,
                                         experimenttracking=experimenttracking,
                                         dataset_type='test',
                                         name=f'{model_name}_r{iteration}_eval_test')
            metrics.update(test_metrics)  # Add or update test metrics to train metrics

    # We trained a new model, maybe derived from a parent model, record its hash
    # Otherwise if model hasn't been trained, no parent relationship is established as it is effectively the same model
    parent_model_hash = framework.hash_model(model_path) if train and model_path is not None else None

    # Do not save loaded model that hasn't been retrained
    if train or not load:

        model_path = framework.export(new_model, f'{model_name}_r{iteration}')

    # Hash new model
    model_hash = framework.hash_model(model_path) if model_path is not None else None

    return TrainResult(name=model_name,
                       i=iteration,
                       model=new_model,
                       params=framework.n_params(new_model),
                       mem_params=framework.n_params(new_model) * 4, # Non-quantized model is assumed to be 32 bits
                       acc=metrics.get('test_acc', None),
                       metrics=metrics,
                       datamodel=datamodel,
                       trainset=datamodel.sets.train,
                       testset=datamodel.sets.test,
                       framework=framework,
                       batch_size=batch_size,
                       optimizer=optimizer,
                       log=True,
                       dataaugmentations=dataaugmentations,
                       experimenttracking=experimenttracking,
                       model_hash=model_hash,
                       parent_model_hash=parent_model_hash,
                       )

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


def get_deployer(model_kind,
                 deploy_target: str,
                 deployers: ModuleType | None = None,
                 deployer_params: DeployerConfigDict | None = None) -> Deployer:
    deployer_params = deployer_params if deployer_params is not None else {}

    if not deployers:  # no custom deployers passed as parameter, check if model suggests custom converter
        converter = getattr(model_kind, 'converter', False)
        if converter and converter.deployers:  # Converter suggested deployers
            deployers = converter.deployers

    return getattr(deployers, deploy_target)(**deployer_params)


def deploy(model_kind: ModuleType,
           deploy_target: str,
           tag: str = 'main',
           deployers: ModuleType | None = None,
           deployer_params: DeployerConfigDict | None = None) -> Deploy | None:
    return get_deployer(model_kind=model_kind,
                        deploy_target=deploy_target,
                        deployers=deployers,
                        deployer_params=deployer_params).deploy(tag=tag)


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
