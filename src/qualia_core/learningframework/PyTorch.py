from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any, Callable, NoReturn

import numpy as np
import numpy.typing
import torch
import torch.distributed
import torch.utils.data
from torch import nn

from qualia_core.experimenttracking.pytorch.ExperimentTrackingPyTorch import ExperimentTrackingPyTorch
from qualia_core.typing import TYPE_CHECKING
from qualia_core.utils.logger import TextLogger

from .LearningFramework import LearningFramework

if TYPE_CHECKING:
    from pytorch_lightning.loggers import Logger  # noqa: TCH002
    from pytorch_lightning.trainer.connectors.accelerator_connector import _PRECISION_INPUT  # noqa: TCH002

    from qualia_core.dataaugmentation.DataAugmentation import DataAugmentation  # noqa: TCH001
    from qualia_core.dataaugmentation.pytorch.DataAugmentationPyTorch import DataAugmentationPyTorch  # noqa: TCH001
    from qualia_core.datamodel.RawDataModel import RawData  # noqa: TCH001
    from qualia_core.experimenttracking.ExperimentTracking import ExperimentTracking  # noqa: TCH001
    from qualia_core.typing import OptimizerConfigDict

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)

class PyTorch(LearningFramework[nn.Module]):
    # Reference framework-specific external modules
    from pytorch_lightning import LightningModule

    import qualia_core.dataaugmentation.pytorch
    import qualia_core.experimenttracking.pytorch
    import qualia_core.learningmodel.pytorch

    dataaugmentations = qualia_core.dataaugmentation.pytorch
    experimenttrackings = qualia_core.experimenttracking.pytorch
    learningmodels = qualia_core.learningmodel.pytorch

    trainer = None

    class TrainerModule(LightningModule):
        def __init__(self,  # noqa: PLR0913
                     model: nn.Module,
                     max_epochs: int = 0,
                     optimizer: OptimizerConfigDict | None = None,
                     dataaugmentations: list[DataAugmentationPyTorch] | None = None,
                     num_classes: int | None = None,
                     experimenttracking_init: Callable[[], NoReturn] | None = None) -> None:
            super().__init__()
            self.model = model
            self.max_epochs = max_epochs
            self.optimizer = optimizer
            self.dataaugmentations = dataaugmentations if dataaugmentations is not None else []
            self.experimenttracking_init = experimenttracking_init

            self.configure_loss()
            self.configure_metrics(num_classes=num_classes)

        @override
        def setup(self, stage: str) -> None:
            super().setup(stage)

            if stage == 'fit':
                if '16' in self.trainer.precision:
                    torch.set_float32_matmul_precision('medium')
                else:
                    torch.set_float32_matmul_precision('high')

            # Required in some cases with ddp_spawn to connect current process with experimenttracking task
            if self.experimenttracking_init is not None:
                self.experimenttracking_init()

        def configure_loss(self) -> None:
            from qualia_core.dataaugmentation.pytorch import Mixup

            self.crossentropyloss = nn.CrossEntropyLoss()
            self.softmax = nn.Softmax(dim=1)

            self.mixup = next((da for da in self.dataaugmentations if isinstance(da, Mixup)), None) # Check if Mixup is enabled
            if self.mixup:
                self.loss = self.mixup.loss.__get__(self.mixup)
            else:
                self.loss = self.crossentropyloss

        def configure_metrics(self, num_classes: int | None = None) -> None:
            import torchmetrics

            metrics = torchmetrics.MetricCollection({
                'prec': torchmetrics.Precision(task='multiclass', average='macro', num_classes=num_classes),
                'rec': torchmetrics.Recall(task='multiclass', average='macro', num_classes=num_classes),
                'f1': torchmetrics.F1Score(task='multiclass', average='macro', num_classes=num_classes),
                'acc': torchmetrics.Accuracy(task='multiclass', average='micro', num_classes=num_classes),
                'avgclsacc': torchmetrics.Accuracy(task='multiclass', average='macro', num_classes=num_classes)})

            self.train_metrics = metrics.clone(prefix='train')
            self.val_metrics = metrics.clone(prefix='val')
            self.test_metrics = metrics.clone(prefix='test')

        @override
        def on_before_batch_transfer(self,
                                    batch: tuple[torch.Tensor, torch.Tensor],
                                    dataloader_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
            if self.dataaugmentations:
                for da in self.dataaugmentations:
                    if (self.trainer.training or da.evaluate) and da.before:
                        batch = da(batch, device=self.device)
            return batch

        @override
        def on_after_batch_transfer(self,
                                    batch: tuple[torch.Tensor, torch.Tensor],
                                    dataloader_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
            if self.dataaugmentations:
                for da in self.dataaugmentations:
                    if (self.trainer.training or da.evaluate) and da.after:
                        batch = da(batch, device=self.device)
            return batch

        @override
        def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_nb: int) -> torch.Tensor:
            x, y = batch
            logits = self(x)

            if not self.mixup:
                self.train_metrics(self.softmax(logits), y)
                self.log_dict(self.train_metrics)

            loss = self.loss(logits, y)
            self.log('train_loss',loss)
            return loss

        @override
        def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_nb: int) -> None:
            x, y = batch
            logits = self.softmax(self(x)) # lightning 1.2 requires preds in [0, 1]

            self.val_metrics(logits, y)
            self.log_dict(self.val_metrics, prog_bar=True)

        @override
        def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_nb: int) -> None:
            x, y = batch
            logits = self.softmax(self(x)) # lightning 1.2 requires preds in [0, 1]

            self.test_metrics(logits, y)
            self.log_dict(self.test_metrics, prog_bar=True)

        @override
        def predict_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
            x, y = batch
            preds = self(x)

            ### Below is boilerplate code to support distributed inference if needed ###
            # By default this will not be used since we set devices=1 in Trainer of PyTorch LearningFramework test()
            shape = torch.tensor(preds.shape, device=preds.device)
            # Fetch shapes of predictions across all nodes
            all_shapes: torch.Tensor = self.all_gather(shape)
            # Apparently only one shape gathered, most likely running on single node so no preds to gather
            if torch.equal(shape, all_shapes):
                return preds

            # Resize preds to fit largest shape since preds will be gathered in a single tensor in a new dim
            # Will be resized back after gathering
            max_shape: torch.Tensor = torch.max(all_shapes, dim=0)[0]
            preds.resize_(*max_shape)

            # Fetch predictions across all nodes
            all_preds: torch.Tensor = self.all_gather(preds)
            # Split to list among the new dimension (1 row per node)
            preds_list = torch.split(all_preds, 1)
            # Resize each preds tensor to its original size
            for shape, t in zip(all_shapes, preds_list):
                t.resize_(*shape)
            # Concat in the same dimension
            return torch.cat(preds_list)

        @override
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x)

        @override
        def on_train_epoch_end(self) -> None:
            super().on_train_epoch_end()

            if not self.mixup:
                self.log_dict(self.train_metrics, prog_bar=True, sync_dist=True)
            for param_group in self.trainer.optimizers[0].param_groups:
                self.log('lr', param_group['lr'], prog_bar=True, sync_dist=True)

            # New line after each train epoch if no validation step
            if not self.trainer.val_dataloaders:
                print()

        @override
        def on_validation_epoch_end(self) -> None:
            super().on_validation_epoch_end()
            # New line after validation step at the end of each epoch
            print()

        @override
        def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer] | list[torch.optim.lr_scheduler.LRScheduler]]:
            import torch.optim as optimizers
            if importlib.util.find_spec('torch_optimizer') is None:
                print('Warning: torch_optimizer not found, not importing additional optimizers', file=sys.stderr)
            else:
                optimizers.__dict__.update(importlib.import_module('torch_optimizer').__dict__) # Merge additional torch_optimizer
            import torch.optim.lr_scheduler as schedulers

            if not self.optimizer:
                return None

            optimizer = getattr(optimizers, self.optimizer['kind'])(self.parameters(), **self.optimizer.get('params', {}))
            print('Optimizer:', optimizer, self.optimizer.get('params', {}))
            if 'scheduler' in self.optimizer:
                try:
                    scheduler = getattr(schedulers, self.optimizer['scheduler']['kind'])(optimizer, **self.optimizer['scheduler']['params'])
                except AttributeError:
                    import qualia_core.learningframework.CustomScheduler as customschedulers
                    scheduler = getattr(customschedulers, self.optimizer['scheduler']['kind'])(optimizer, **self.optimizer['scheduler']['params'])

                print('Scheduler:', scheduler, self.optimizer['scheduler']['params'])
                return [optimizer], [scheduler]
            return [optimizer], []

    def __init__(self,
                 use_best_epoch: bool = False,
                 enable_progress_bar: bool = True,
                 progress_bar_refresh_rate: int = 1,
                 accelerator: str = 'auto',
                 devices: int | str | list[int] = 'auto',
                 precision: _PRECISION_INPUT = 32) -> None:
        super().__init__()
        self._use_best_epoch = use_best_epoch
        self._enable_progress_bar = enable_progress_bar
        self._progress_bar_refresh_rate = progress_bar_refresh_rate
        self.accelerator = accelerator
        self.devices = devices
        self.precision = precision

        self.log = TextLogger(name=__name__)

        # Force using 'spawn' instead of 'popen' start_method for ddp strategy in case of multiple devices.
        # This is done to avoid starting our 'qualia' script all over again which causes some issues
        # since it does much more than just perform the training, this would result in all the steps being performed again
        # including duplicating tests, postprocessing, etc…
        # This is probably significantly slower than using 'popen' (and may even lead to longer run time than single device)
        # but at least it allows running multi-GPU training by default, and running single device exclusively can still be done by
        # settings learningframework.devices to 1 or exposing only one device with CUDA_VISIBLE_DEVICES.
        # We modify the registry so that we can still use strategy='auto' in the trained to use the SingleDevice strategy in case
        # only one device is being used since this is much faster than initializing DDP.
        from pytorch_lightning.strategies import StrategyRegistry
        from pytorch_lightning.strategies.ddp import DDPStrategy
        start_method = 'spawn'
        StrategyRegistry.register('ddp',
                                   DDPStrategy,
                                   description=f'DDP strategy with `start_method={start_method!r}`',
                                   start_method='spawn',
                                   override=True)

    @staticmethod
    def channels_last_to_channels_first(x: numpy.typing.NDArray[Any]) -> numpy.typing.NDArray[Any]:
        if len(x.shape) == 4:
            x = x.transpose(0, 3, 1, 2)
        elif len(x.shape) == 3:
            x = x.swapaxes(1, 2)
        else:
            raise ValueError(f'Unsupported number of axes in dataset: {len(x.shape)}, must be 3 or 4')
        return x

    @staticmethod
    def channels_first_to_channels_last(x: numpy.typing.NDArray[Any]) -> numpy.typing.NDArray[Any]:
        if len(x.shape) == 4:
            x = x.transpose(0, 2, 3, 1)
        elif len(x.shape) == 3:
            x = x.swapaxes(2, 1)
        else:
            raise ValueError(f'Unsupported number of axes in dataset: {len(x.shape)}, must be 3 or 4')
        return x


    class DatasetFromArray(torch.utils.data.Dataset[tuple[numpy.typing.NDArray[np.float32], numpy.typing.NDArray[np.int32]]]):
        def __init__(self, dataset: RawData) -> None:
            super().__init__()
            self.x = PyTorch.channels_last_to_channels_first(dataset.x)
            self.y = dataset.y.argmax(axis=-1)
            self.y = np.where(np.isnan(dataset.y).any(axis=-1),
                -1,
                self.y)

        def __len__(self) -> int:
            return len(self.x)

        @override
        def __getitem__(self, index: int) -> tuple[numpy.typing.NDArray[np.float32], numpy.typing.NDArray[np.int32]]:
            return self.x[index], self.y[index]

    def logger(self,
               experimenttracking: ExperimentTracking | None,
               name: str) -> list[Logger]:
        from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
        loggers: list[Logger] = [CSVLogger(save_dir='logs/PyTorchLightning', name=name)]
        if importlib.util.find_spec('tensorboard') is None and importlib.util.find_spec('tensorboardX') is None:
            logger.warning('tensorboard or tensorboardX not found, disabling TensorBoardLogger')
        else:
            loggers.append(TensorBoardLogger(save_dir='lightning_logs', name=name))
        if isinstance(experimenttracking, ExperimentTrackingPyTorch) and experimenttracking.logger is not None:
            loggers.append(experimenttracking.logger)
        return loggers

    @override
    def train(self,  # noqa: PLR0913
              model: nn.Module,
              trainset: RawData | None,
              validationset: RawData | None,
              epochs: int,
              batch_size: int,
              optimizer: OptimizerConfigDict | None,
              dataaugmentations: list[DataAugmentation] | None = None,
              experimenttracking: ExperimentTracking | None = None,
              name: str | None = None,
              precision: _PRECISION_INPUT | None = None) -> nn.Module:
        import os

        from pytorch_lightning import Trainer, seed_everything
        from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
        from torch.utils.data import DataLoader

        # PyTorch-Lightning >= 1.3.0 resets seed before training, increment seed between trainings to get different values between experiments
        seed = os.environ.get("PL_GLOBAL_SEED", None)
        if seed is None:
            print("Warning: PyTorch not seeded", file=sys.stderr)
        else:
            seed_everything(int(seed) * 100)

        checkpoint_callback = ModelCheckpoint(dirpath=f"out/checkpoints/{name}", save_top_k=2, monitor="valavgclsacc", mode="max")
        callbacks = [checkpoint_callback]
        if self._enable_progress_bar:
            callbacks.append(TQDMProgressBar(refresh_rate=self._progress_bar_refresh_rate))

        experimenttracking_init = experimenttracking.initializer if experimenttracking is not None else None

        trainer = Trainer(max_epochs=epochs,
                          accelerator=self.accelerator,
                          devices=self.devices,
                          precision=self.precision if precision is None else precision,
                          deterministic=True,
                          logger=self.logger(experimenttracking, name=name),
                          enable_progress_bar=self._enable_progress_bar,
                          callbacks=callbacks)
        trainer_module = self.TrainerModule(model,
                                            max_epochs=epochs,
                                            optimizer=optimizer,
                                            dataaugmentations=dataaugmentations,
                                            num_classes=trainset.y.shape[-1],
                                            experimenttracking_init=experimenttracking_init)
        #self.trainer.fit(trainer_module,
        #                    DataLoader(self.DatasetFromTF(trainset), batch_size=None), [
        #                        DataLoader(self.DatasetFromTF(originalset), batch_size=None),
        #                        DataLoader(self.DatasetFromTF(validationset), batch_size=None)
        #                    ])
        # Bug in PyTorch-Lightning 1.0 with multiple dataloaders

        # Do not attempt to train with no epochs
        if epochs < 1:
            return model

        print('Epochs:', epochs, trainer.max_epochs)
        trainer.fit(trainer_module,
                            DataLoader(self.DatasetFromArray(trainset), batch_size=batch_size,shuffle=True),
                            #num_workers=2, persistent_workers=True, pin_memory=True),
                            DataLoader(self.DatasetFromArray(validationset), batch_size=batch_size) if validationset is not None else None
                        )

        if self._use_best_epoch:
            print(f'Loading back best epoch: {checkpoint_callback.best_model_path}, score: {checkpoint_callback.best_model_score}')
            trainer_module = self.TrainerModule.load_from_checkpoint(checkpoint_callback.best_model_path,
                                                                    model=model,
                                                                    optimizer=optimizer,
                                                                    dataaugmentations=dataaugmentations,
                                                                    num_classes=trainset.y.shape[-1])
            model = trainer_module.model
        return model

    @override
    def evaluate(self,
                 model: nn.Module,
                 testset: RawData,
                 batch_size: int,
                 dataaugmentations: list[DataAugmentation],
                 experimenttracking: ExperimentTracking | None = None,
                 dataset_type: str = '',
                 name: str = '') -> dict[str, int | float | numpy.typing.NDArray[Any]]:
        import torch
        from pytorch_lightning import Trainer
        from pytorch_lightning.callbacks import TQDMProgressBar
        from torch.utils.data import DataLoader

        self.log(f'{name=}')
        self.log(f'{dataset_type=}')

        # Force testing on single device to avoid issues caused by DDP generating different batches
        # Either first of given devices list or first default
        devices = self.devices[:1] if isinstance(self.devices, list) else 1

        callbacks = []
        if self._enable_progress_bar:
            callbacks.append(TQDMProgressBar(refresh_rate=self._progress_bar_refresh_rate))

        trainer = Trainer(max_epochs=0,
                          accelerator=self.accelerator,
                          devices=devices,
                          deterministic=True,
                          logger=self.logger(experimenttracking, name=name),
                          enable_progress_bar=self._enable_progress_bar,
                          callbacks=callbacks)
        trainer_module = self.TrainerModule(model, dataaugmentations=dataaugmentations, num_classes=testset.y.shape[-1])
        metrics = trainer.test(trainer_module, DataLoader(self.DatasetFromArray(testset), batch_size=batch_size))
        predictions = torch.cat(trainer.predict(trainer_module, DataLoader(self.DatasetFromArray(testset), batch_size=batch_size)))
        # predict returns list of predictions according to batches
        self.log(f'{metrics=}')

        print('Confusion matrix:')
        cm = self.confusion_matrix(predictions, testset, device=trainer_module.device)
        ncm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        avg_class_accuracy = ncm.diagonal().mean()
        print(f'{avg_class_accuracy=}')
        self.log(f'{avg_class_accuracy=}')

        with np.printoptions(threshold=sys.maxsize, suppress=True, linewidth=sys.maxsize, precision=2):
            print(cm)
            print("Normalized:")
            print(ncm)

            self.log(f'{cm=}')
            self.log(f'{ncm=}')

            if experimenttracking is not None and experimenttracking.logger is not None:
                experimenttracking.logger.experiment['cm'].log(np.array2string(cm))
                experimenttracking.logger.experiment['ncm'].log(np.array2string(ncm))
        metrics[0]['cm'] = cm
        metrics[0]['ncm'] = ncm

        return metrics[0]

    @override
    def predict(self,  # noqa: PLR0913
                 model: nn.Module,
                 dataset: RawData,
                 batch_size: int,
                 dataaugmentations: list[DataAugmentation],
                 experimenttracking: ExperimentTracking | None = None,
                 name: str = '') -> torch.Tensor:
        from pytorch_lightning import Trainer
        from pytorch_lightning.callbacks import Callback, TQDMProgressBar
        from torch.utils.data import DataLoader

        # Force testing on single device to avoid issues caused by DDP generating different batches
        # Either first of given devices list or first default
        devices = self.devices[:1] if isinstance(self.devices, list) else 1

        callbacks: list[Callback] = []
        if self._enable_progress_bar:
            callbacks.append(TQDMProgressBar(refresh_rate=self._progress_bar_refresh_rate))

        trainer = Trainer(max_epochs=0,
                          accelerator=self.accelerator,
                          devices=devices,
                          deterministic=True,
                          logger=self.logger(experimenttracking, name=name),
                          enable_progress_bar=self._enable_progress_bar,
                          callbacks=callbacks)
        trainer_module = self.TrainerModule(model,
                                            dataaugmentations=dataaugmentations,
                                            num_classes=dataset.y.shape[-1])
        predictions = trainer.predict(trainer_module,
                               DataLoader(self.DatasetFromArray(dataset), batch_size=batch_size))

        if predictions is None:
            logger.error('predict() returned None')
            raise RuntimeError
        if not isinstance(predictions[0], torch.Tensor):
            logger.error('Expected predict() result to be a list of torch.Tensor, got: %s', type(predictions[0]))
            raise TypeError

        return torch.cat(predictions)

    def confusion_matrix(self, predictions: torch.Tensor, testset: RawData, device: str) -> numpy.typing.NDArray[np.int32]:
        import torch
        from torchmetrics import ConfusionMatrix

        testy = torch.tensor(testset.y.argmax(axis=1), device=device)

        confmat = ConfusionMatrix(task='multiclass', num_classes=testset.y.shape[1])
        return confmat(predictions.to(device=device).argmax(axis=1), testy).int().numpy()

    @override
    def load(self, name: str, model: nn.Module) -> nn.Module:
        import torch
        path = Path('out')/'learningmodel'/f'{name}.pth'
        if path.is_file():
            state_dict = torch.load(path)
            model.load_state_dict(state_dict)
            logger.info('Loaded %s.', path)
        else:
            logger.warning('%s not found, not loading weights.', path)
        return model

    @override
    def export(self, model: nn.Module, name: str) -> None:
        import torch
        outdir = Path('out')/'learningmodel'
        outdir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), outdir/f'{name}.pth')

    @override
    def summary(self, model: nn.Module) -> None:
        """Print model summary."""
        print(model)
        print(f'Number of parameters: {self.n_params(model)}')

    @override
    def n_params(self, model: nn.Module) -> int:
        import numpy as np
        return np.sum([params.numel() for params in model.parameters()])

    @override
    def save_graph_plot(self, model: nn.Module, model_save: str) -> None:
        import sys
        print('Warning: saving PyTorch graph plot unsupported', file=sys.stderr)

    @override
    def apply_dataaugmentation(self,
                               da: DataAugmentationPyTorch,
                               x: numpy.typing.NDArray[Any],
                               y: numpy.typing.NDArray[Any],
                               device: torch.device | None = None) -> tuple[numpy.typing.NDArray[Any], numpy.typing.NDArray[Any]]:
        import torch
        x = self.channels_last_to_channels_first(x)
        tensor_x = torch.tensor(x, device=device)
        tensor_y = torch.tensor(y, device=device)
        tensor_x, tensor_y = da((tensor_x, tensor_y), device=device)
        x = tensor_x.numpy()
        y = tensor_y.numpy()
        x = self.channels_first_to_channels_last(x)
        return x, y
