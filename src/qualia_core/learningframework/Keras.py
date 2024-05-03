from __future__ import annotations

import sys
from pathlib import Path

import keras  # type: ignore[import-untyped]

from qualia_core.typing import TYPE_CHECKING

from .LearningFramework import LearningFramework

if TYPE_CHECKING:
    import tensorflow as tf  # type: ignore[import-untyped] # noqa: TCH002

    from qualia_core.dataaugmentation.DataAugmentation import DataAugmentation  # noqa: TCH001
    from qualia_core.datamodel.RawDataModel import RawData  # noqa: TCH001
    from qualia_core.experimenttracking.ExperimentTracking import ExperimentTracking  # noqa: TCH001

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class Keras(LearningFramework[keras.Model]):
    # Reference framework-specific external modules
    import qualia_core.learningmodel.keras as learningmodels
    import qualia_core.experimenttracking.pytorch as experimenttrackings

    def __init__(self):
        import tensorflow as tf
        # Cleanup, especially reset graph uid for uniquely generated layer names
        tf.keras.backend.clear_session()

    def __configure_optimizers(self, optimizer=None):
        import tensorflow as tf

        if optimizer is None:
            from .DummyOptimizer import DummyOptimizer
            return DummyOptimizer()

        opt = getattr(tf.keras.optimizers, optimizer['kind'])
        if 'scheduler' in optimizer:
            scheduler = getattr(tf.keras.optimizers.schedules, optimizer['scheduler']['kind'])(**optimizer['scheduler'].get('params', {}))
            opt = opt(learning_rate=scheduler, **optimizer.get('params', {}))
            print('Scheduler:', scheduler, optimizer['scheduler'].get('params', {}))
        else:
            opt = opt(**optimizer.get('params', {}))
        print('Optimizer:', opt, optimizer.get('params', {}))
        return opt


    def train(self,
              model,
              trainset,
              validationset,
              epochs,
              batch_size,
              optimizer,
              dataaugmentations=None,
              experimenttracking=None,
              name: str='') -> keras.Model:
        # gpus handled by TensorFlowInitializer
        import tensorflow as tf

        if dataaugmentations:
            raise ValueError(f'Data augmentation not supported with {self.__class__.__name__}')

        # You can plot the quantize training graph on tensorboard
        logdir='out/logs/fit/{name}/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

        callbacks=[]
        #callbacks.append(tensorboard_callback)
        # The patience parameter is the amount of epochs to check for improvement
        #callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5))
        #callbacks.append(tfdocs.modeling.EpochDots())
        if experimenttracking is not None:
            callbacks.append(experimenttracking.callback)

        optimizer = self.__configure_optimizers(optimizer)

        accuracy_metric = tf.keras.metrics.CategoricalAccuracy(name='trainacc')
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[accuracy_metric])

        model.fit(trainset.x,
                trainset.y,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=validationset.astuple() if validationset is not None else None,
                callbacks=callbacks)

        # Remove softmax
        #compile_params = {'loss': model.loss, 'optimizer': model.optimizer, 'metrics': model.compiled_metrics.metrics}
        #model = tf.keras.Model(model.input, model.layers[-2].output, name=model.name)
        #model.compile(**compile_params)
        #model.summary()

        return model

    def load(self, name, model=None, path: Path=Path('out')/'learningmodel'):
        from tensorflow.keras.models import load_model
        model = load_model(path/f'{name}.h5', custom_objects=getattr(model, 'get_custom_objects', lambda: {})())
        return model

    def evaluate(self, model, testset, batch_size, gpus: int=None, dataaugmentations=None, experimenttracking=None, dataset_type: str='', name: str=''):
        import tensorflow as tf

        if dataaugmentations:
            raise ValueError(f'Data augmentation not supported with {self.__class__.__name__}')

        accuracy_metric = tf.keras.metrics.CategoricalAccuracy(name='testacc')
        model.compile(loss='categorical_crossentropy', metrics=[accuracy_metric])

        metrics = model.evaluate(testset.x, testset.y, batch_size=batch_size)
        metrics_names = model.metrics_names

        # Workaround yet another Keras 3.x nonsense
        try:
            from keras.src.trainers.compile_utils import CompileMetrics

            metrics_names = []
            for m in model.metrics:
                if isinstance(m, CompileMetrics):
                    for mm in m.metrics:
                        metrics_names.append(mm.name)
                else:
                    metrics_names.append(m.name)

        except ModuleNotFoundError:
            pass

        # Predict and show confusion matrix
        train_predictions = model.predict(testset.x)

        cm = tf.math.confusion_matrix(testset.y.argmax(axis=1), train_predictions.argmax(axis=1))
        print(cm)
        if experimenttracking is not None:
            experimenttracking.logger.experiment['cm'].log(np.array2string(cm))

        metrics = {name: metric for name, metric in zip(metrics_names, metrics)}
        metrics['cm'] = cm

        return metrics

    @override
    def predict(self,  # noqa: PLR0913
                 model: keras.Model,
                 dataset: RawData,
                 batch_size: int,
                 dataaugmentations: list[DataAugmentation],
                 experimenttracking: ExperimentTracking | None,
                 name: str) -> tf.Tensor:
        raise NotImplementedError

    def export(self, model, name, path: Path=Path('out')/'learningmodel'):
        model.save(path/f'{name}.h5')

    def summary(self, model):
        model.summary()

    def n_params(self, model):
        return model.count_params()

    def save_graph_plot(self, model, model_save):
        import tensorflow as tf
        tf.keras.utils.model_to_dot(model, show_shapes=True, expand_nested=True).write(f'{model_save}.dot')

    def apply_dataaugmentation(self):
        raise NotImplemented()
