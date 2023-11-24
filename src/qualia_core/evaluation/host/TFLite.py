from pathlib import Path
import re
import sys
import time
from qualia_core.evaluation.Stats import Stats
import numpy as np

class TFLite:
    def __init__(self, *args, **kwargs):
        pass

    def evaluate(self, framework, model_kind, dataset, target: str, tag: str, limit: int=None, dataaugmentations=[]):
        if dataaugmentations:
            raise ValueError(f'dataaugmentations not supported for {self.__class__.__name__}')

        import tensorflow as tf
        from sklearn.metrics import accuracy_score

        interpreter = tf.lite.Interpreter(model_path=str(Path('out')/'deploy'/'tflite'/f'{tag}.tflite'))
        interpreter.allocate_tensors()
        pred_test_q = np.ndarray(dataset.sets.test.y.shape)

        tstart = time.time() # Start timer
        for i, x in enumerate(dataset.sets.test.x):
            interpreter.set_tensor(interpreter.get_input_details()[0]['index'], [x])
            interpreter.invoke()
            pred_test_q[i] = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
        tstop = time.time() # Stop timer

        accuracy = accuracy_score(dataset.sets.test.y.argmax(axis=1), pred_test_q.argmax(axis=1))

        print('Quantized model accuracy: ', accuracy)
        print(tf.math.confusion_matrix(dataset.sets.test.y.argmax(axis=1), pred_test_q.argmax(axis=1)))

        return Stats(avg_time=(tstop - tstart) / len(dataset.sets.test.y), accuracy=accuracy)
