from __future__ import annotations

import copy
import dataclasses
import logging
import sys
from collections.abc import Generator
from pathlib import Path
from typing import Any, Callable, NamedTuple, cast

from torch import nn

from qualia_core.learningframework.PyTorch import PyTorch
from qualia_core.typing import TYPE_CHECKING, ModelConfigDict, ModelParamsConfigDict, OptimizerConfigDict, RecursiveConfigDict
from qualia_core.utils.logger import CSVLogger
from qualia_core.utils.merge_dict import merge_dict

from .PostProcessing import PostProcessing

if TYPE_CHECKING:
    import keras  # type: ignore[import-untyped]  # noqa: TCH002
    import numpy.typing  # noqa: TCH002

    from qualia_core.datamodel.RawDataModel import RawData  # noqa: TCH001
    from qualia_core.learningframework.LearningFramework import LearningFramework, T  # noqa: TCH001
    from qualia_core.qualia import TrainResult  # noqa: TCH001

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)

class QuantizationAwareTrainingLoggerFields(NamedTuple):
    i: int
    name: str
    step: str
    width: int
    metrics: dict[str, int | float | numpy.typing.NDArray[Any]]

class QuantizationAwareTraining(PostProcessing[nn.Module]):

    def __init__(self,  # noqa: PLR0913
                 epochs: int = 1,
                 batch_size: int = 32,
                 model: RecursiveConfigDict | None = None,
                 optimizer: OptimizerConfigDict | None = None,
                 evaluate_before: bool = True) -> None:  # noqa: FBT002, FBT001
        super().__init__()

        self.qmodel = model if model is not None else {}
        self.epochs = epochs
        self.batch_size = batch_size
        self.qoptimizer = optimizer
        self._evaluate_before = evaluate_before

        if 'params' not in self.qmodel or not isinstance(self.qmodel['params'], dict):
            logger.error("'model.params' is required to be a table")
            raise ValueError
        if 'quant_params' not in self.qmodel['params'] or not isinstance(self.qmodel['params']['quant_params'], dict):
            logger.error("'model.params.quant_params' is required to be a table")
            raise ValueError
        self.qmodel_params: ModelParamsConfigDict = self.qmodel['params'] # TODO: validate

        quant_params = self.qmodel_params['quant_params']

        self.width: int = quant_params.get('bits', 0)
        self.force_q: int | None = quant_params.get('force_q', None)
        self.lsq: bool = quant_params.get('LSQ', False)

        self.log: CSVLogger[QuantizationAwareTrainingLoggerFields] = CSVLogger(name=__name__)
        self.log.fields = QuantizationAwareTrainingLoggerFields

    def __evaluate(self,  # noqa: PLR0913
                   framework: LearningFramework[T],
                   model: T,
                   trainset: RawData,
                   testset: RawData,
                   trainresult: TrainResult,
                   step: str) -> tuple[dict[str, int | float | numpy.typing.NDArray[Any]], str]:
        logger.info('Evaluation on train dataset')
        name = f'{trainresult.name}_q{self.width}_r{trainresult.i}_{step}_eval_train'
        metrics = framework.evaluate(model,
                                     trainset,
                                     batch_size=self.batch_size,
                                     dataaugmentations=trainresult.dataaugmentations,
                                     experimenttracking=trainresult.experimenttracking,
                                     dataset_type='train',
                                     name=name)

        if len(testset.x) > 0: # Don't evaluate if testset is empty
            logger.info('Evaluation on test dataset')
            name = f'{trainresult.name}_q{self.width}_r{trainresult.i}_{step}_eval_test'
            metrics = framework.evaluate(model,
                                         testset,
                                         batch_size=self.batch_size,
                                         dataaugmentations=trainresult.dataaugmentations,
                                         experimenttracking=trainresult.experimenttracking,
                                         dataset_type='test',
                                         name=name)
        return metrics, name

    def __evaluate_and_log(self,  # noqa: PLR0913
                           framework: LearningFramework[T],
                           model: T,
                           trainset: RawData,
                           testset: RawData,
                           trainresult: TrainResult,
                           step: str,
                           step_msg: str) -> dict[str, int | float | numpy.typing.NDArray[Any]]:
        logger.info('Evaluation %s', step_msg)
        metrics, name = self.__evaluate(framework,
                                        model,
                                        trainset,
                                        testset,
                                        trainresult=trainresult,
                                        step=step)
        self.log(QuantizationAwareTrainingLoggerFields(i=trainresult.i,
                                                       name=name,
                                                       step=step,
                                                       width=self.width,
                                                       metrics=metrics))

        return metrics

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

        prefix = 'Quantized'
        if self.quant_framework is not None :
            prefix = prefix + self.quant_framework
            if self.quant_framework == 'brevitas' :
                import torch
                torch.use_deterministic_algorithms(mode=True)

        # Use model name from config prefixed with Quantized
        quantized_model: nn.Module = getattr(framework.learningmodels, prefix + model_conf['kind'])(**qmodel_params)
        _ = quantized_model.load_state_dict(model.state_dict(), strict=True)
        framework.summary(quantized_model)

        return quantized_model, qmodel_params

    def _update_activation_ranges(self,
                                  framework: PyTorch,
                                  trainresult: TrainResult,
                                  quantized_model: nn.Module) -> nn.Module:
        logger.info('Updating activation ranges')
        name = f'{trainresult.name}_q{self.width}_r{trainresult.i}_update_activation_ranges'
        # Must train to update activation range
        return framework.train(quantized_model,
                        trainset=trainresult.trainset,
                        validationset=trainresult.datamodel.sets.valid,
                        epochs=1,
                        batch_size=self.batch_size,
                        optimizer=None, # Disable optimizer
                        dataaugmentations=trainresult.dataaugmentations,
                        experimenttracking=trainresult.experimenttracking,
                        name=name,
                        precision=32) # Force precision=32 since 'mixed-16' won't work without optimizer, related: https://github.com/Lightning-AI/lightning/issues/17407

    def _quantization_aware_training(self,
                                     framework: LearningFramework[T],
                                     trainresult: TrainResult,
                                     quantized_model: T) -> T:
            logger.info('Performing quantization-aware training for %s epochs', self.epochs)
            # restore optimizer with overwritten params from QAT config
            optimizer = trainresult.optimizer if self.qoptimizer is None else merge_dict(self.qoptimizer, trainresult.optimizer)
            name = f'{trainresult.name}_q{self.width}_r{trainresult.i}_post_train'
            return framework.train(quantized_model,
                            trainset=trainresult.datamodel.sets.train,
                            validationset=trainresult.datamodel.sets.valid,
                            epochs=self.epochs,
                            batch_size=self.batch_size,
                            optimizer=optimizer,
                            dataaugmentations=trainresult.dataaugmentations,
                            experimenttracking=trainresult.experimenttracking,
                            name=name)

    def __export_activation_ranges(self,
                                   model: nn.Module,
                                   save_name: str) -> None:
        from torch.fx.graph import _Namespace, _snake_case

        from qualia_core.learningmodel.pytorch.layers.quantized_layers import quantized_layers
        from qualia_core.learningmodel.pytorch.layers.QuantizedLayer import QuantizedLayer
        from qualia_core.learningmodel.pytorch.Quantizer import Quantizer

        namespace = _Namespace()

        path = Path('out')/'learningmodel'/f'{save_name}_activations_range.txt'
        logger.info('Exporting activation ranges to %s', path)

        with path.open('w') as f:
            for name, m in cast(Generator[tuple[str, nn.Module], None, None], model.named_modules()):
                if len(name) > 0: # _Namespace.create_name tries to access first character without any check for empty string
                    name = _snake_case(name) # convert module name to identifier, '.' replaced with '_', this is what torch.fx uses
                    name = namespace.create_name(name, m)
                if isinstance(m, (QuantizedLayer, *quantized_layers)):
                    q = f'{name},{m.input_q},{m.activation_q},{m.weights_q},{m.bias_q}'
                    q += f',{m.input_round_mode},{m.activation_round_mode},{m.weights_round_mode}'
                    print(q, file=f)
                    logger.info('%s', q)
                elif not isinstance(m, Quantizer): # Skip Quantizer modules as they need no quantization themselves
                    logger.info("Layer '%s' of type '%s' not quantized", name, type(m))

    @override
    def __call__(self, trainresult: TrainResult, model_conf: ModelConfigDict) -> tuple[ TrainResult, ModelConfigDict]:
        framework = trainresult.framework
        if not isinstance(framework, PyTorch):
            logger.error('Framework %s is not compatible', type(framework))
            raise TypeError

        et = trainresult.experimenttracking

        if et is not None:
            et.start(name=f'{trainresult.name}_q{self.width}_r{trainresult.i}')
            et.hyperparameters = {'model_conf': model_conf,
                                  'qmodel': self.qmodel,
                                  'qoptimizer': self.qoptimizer,
                                  'i': trainresult.i}

        if self._evaluate_before:
            _ = self.__evaluate_and_log(framework=framework,
                                    model=trainresult.model,
                                    trainset=trainresult.trainset,
                                    testset=trainresult.testset,
                                    trainresult=trainresult,
                                    step='baseline',
                                    step_msg='before quantization')

        quantized_model, qmodel_params = self._build_quantized_model(model=trainresult.model,
                                                      framework=framework,
                                                      model_conf=model_conf)

        quantized_model = self._update_activation_ranges(framework=framework,
                                       trainresult=trainresult,
                                       quantized_model=quantized_model)

        metrics = self.__evaluate_and_log(framework=framework,
                                model=quantized_model,
                                trainset=trainresult.trainset,
                                testset=trainresult.testset,
                                trainresult=trainresult,
                                step='PTQ',
                                step_msg='after quantization without training (PTQ)')

        if self.epochs > 0: # Do not attempt quantization-aware training if epochs==0
            quantized_model = self._quantization_aware_training(framework=framework,
                                              trainresult=trainresult,
                                              quantized_model=quantized_model)
            metrics = self.__evaluate_and_log(framework=framework,
                                    model=quantized_model,
                                    trainset=trainresult.trainset,
                                    testset=trainresult.testset,
                                    trainresult=trainresult,
                                    step='QAT',
                                    step_msg='after quantization-aware training')

        model_name = self.process_name(trainresult.name)

        self.__export_activation_ranges(quantized_model, save_name=f'{model_name}_r{trainresult.i}')

        if et is not None:
            et.stop()

        new_model_conf = copy.deepcopy(model_conf)
        new_model_conf['params'] = qmodel_params

        return dataclasses.replace(trainresult, model=quantized_model,
                                    name=model_name,
                                    params=framework.n_params(quantized_model),
                                    mem_params=(framework.n_params(quantized_model) * self.width) // 8,
                                    acc=metrics['testacc'],
                                    metrics=metrics), new_model_conf

    @override
    def process_name(self, name: str) -> str:
        return f'{name}_q{self.width}_force_q{self.force_q if self.force_q is not None else "off"}_e{self.epochs}_LSQ{self.lsq}'

    @override
    def process_mem_params(self, mem_params: int) -> Callable[[LearningFramework[nn.Module | keras.Model],
                                                               nn.Module | keras.Model], int]:
        return lambda framework, model: (framework.n_params(model) * self.width) // 8
