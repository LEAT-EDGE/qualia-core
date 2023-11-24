from pathlib import Path
import re
import sys
import time
from qualia_core.evaluation.Stats import Stats
import numpy as np
import multiprocessing

def benchmark(tag, model_kind, dataset, repeat, batch_size, use_gpu, ret):
    from qualia_core.utils import TensorFlowInitializer
    from qualia_core.learningframework import Keras

    tfi = TensorFlowInitializer()
    tfi(reserve_gpu=use_gpu, gpu_memory_growth=False)

    import qualia_core.learningmodel.keras as models
    model_kind = getattr(models, model_kind)

    framework = Keras()
    model = framework.load(tag, model=model_kind, path=Path('out')/'deploy'/'keras')

    print('Started')
    tiledset = np.tile(dataset.sets.test.x, (repeat, 1, 1))
    tstart = time.perf_counter_ns() # Start timer
    train_predictions = model.predict(tiledset, batch_size=batch_size)
    tstop = time.perf_counter_ns() # Stop timer
    avg_time = (tstop - tstart) / (1000000000 * len(tiledset))
    print('Finished')

    ret.value = avg_time

class Keras:
    def __init__(self, batch_size, repeat, use_gpu, *args, **kwargs):
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.repeat = repeat


    def evaluate(self, model_kind, dataset, target: str, tag: str, limit: int=None):
        import tensorflow as tf
        from qualia_core.learningframework import Keras

        framework = Keras()
        model = framework.load(tag, model=model_kind, path=Path('out')/'deploy'/'keras')
        model.compile(metrics=['CategoricalAccuracy'])
        
        results = model.evaluate(dataset.sets.test.x, dataset.sets.test.y, batch_size=self.batch_size, return_dict=True)

        ctx = multiprocessing.get_context('spawn')
        ret = ctx.Value("d", 0.0, lock=False)
        p = ctx.Process(target=benchmark, args=(tag, model_kind, dataset, self.repeat, self.batch_size, self.use_gpu, ret))
        p.start()
        p.join()
        avg_time = ret.value

        train_predictions = model.predict(dataset.sets.test.x)
        print(tf.math.confusion_matrix(dataset.sets.test.y.argmax(axis=1), train_predictions.argmax(axis=1)))

        return Stats(avg_time=avg_time, accuracy=results['categorical_accuracy'])
