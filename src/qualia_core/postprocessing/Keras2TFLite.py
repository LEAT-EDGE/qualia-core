# See https://www.tensorflow.org/lite/performance/post_training_quantization

import functools
import importlib

class Keras2TFLite:
    # TFLite can be deployed both with STM32CubeAI on STM32 boards and TFLiteMicro on other boards (SparkFun Edge…)
    import qualia_core.deployment.tflitemicro as deployers # Use TFLiteMicro deployers
    deployers.__dict__.update(importlib.import_module('qualia_core.deployment.stm32cubeai').__dict__) # Merge STM32CubeAI deployers

    def __init__(self, quantize: int=-1, new_converter=False):
        if quantize not in ['float32', 'float16', 'float8', 'int8', 'int16']:
            raise ValueError(self.__class__.__name__ + ' only suports none, 8 or 16-bit quantization')
        self.__quantize = quantize
        self.__new_converter = new_converter

        if quantize == 'float32':
            self.__width = 32
        elif quantize == 'float16' or quantize == 'int16':
            self.__width = 16
        elif quantize == 'float8' or quantize == 'int8':
            self.__width = 8

    def representative_dataset_gen(self, representative_dataset):
        for input_value in representative_dataset:
            yield [input_value.reshape([1, *input_value.shape])]

    def convert(self, framework, keras_model, model_name, representative_dataset=None):
        import tensorflow as tf
        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        converter.experimental_new_converter = self.__new_converter

        if self.__quantize == 'float16':
            raise ValueError('float16 quantization not supported by TFLite Micro')
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        elif self.__quantize == 'float8':
            raise ValueError('float8 quantization not supported by TFLite Micro')
            converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        elif self.__quantize == 'int8':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = functools.partial(self.representative_dataset_gen, representative_dataset)
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            # We use float inputs/outputs
            #converter.inference_input_type = tf.int8
            #converter.inference_output_type = tf.int8
            converter.inference_input_type = tf.float32
            converter.inference_output_type = tf.float32
        elif self.__quantize == 'int16':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = functools.partial(self.representative_dataset_gen, representative_dataset)
            converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
            #converter.inference_input_type = tf.int16
            #converter.inference_output_type = tf.int16
            # Old converter is buggy for int8/int16 quant, it cannot use float inputs or representative_dataset
            converter.experimental_new_converter = True
        elif self.__quantize != 'float32':
            raise NotImplemented(f'{self.__quantize} no implemented in {self.__class__.__name__}')

        self.__data = converter.convert()
        self.__input_shape = keras_model.input_shape

        return self

    @property
    def data(self) -> bytes:
        return self.__data

    @property
    def input_shape(self):
        return self.__input_shape

    def process_mem_params(self, mem_params):
        return lambda framework, model: (framework.n_params(model) * self.__width) // 8
