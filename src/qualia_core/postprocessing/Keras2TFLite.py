# See https://www.tensorflow.org/lite/performance/post_training_quantization

from __future__ import annotations

import functools
import logging
import sys
import typing
from typing import Any, Literal

import keras  # type: ignore[import-untyped] # No stubs for keras package
import tensorflow as tf  # type: ignore[import-untyped] # No stubs for keras package

import qualia_core.deployment.stm32cubeai
import qualia_core.deployment.tflitemicro
from qualia_core.typing import TYPE_CHECKING

from .Converter import Converter

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Generator

if TYPE_CHECKING:
    if sys.version_info >= (3, 10):
        from typing import TypeGuard
    else:
        from typing_extensions import TypeGuard


    import numpy.typing  # noqa: TCH002

    from qualia_core.learningframework.LearningFramework import LearningFramework  # noqa: TCH001

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

QuantizeType = Literal['float32', 'float16', 'float8', 'int16', 'int8']
logger = logging.getLogger(__name__)

class Keras2TFLite(Converter[keras.Model]):
    # TFLite can be deployed both with STM32CubeAI on STM32 boards and TFLiteMicro on other boards (SparkFun Edge…)
    deployers = qualia_core.deployment.tflitemicro # Use TFLiteMicro deployers
    deployers.__dict__.update(qualia_core.deployment.stm32cubeai.__dict__) # Merge STM32CubeAI deployers

    __quantize: QuantizeType
    __input_shape: tuple[int, ...]
    __data: bytes

    def __init__(self,
                 quantize: str,
                 new_converter: bool = False) -> None:  # noqa: FBT001, FBT002
        super().__init__()

        if not self.__check_quantize(quantize):
            logger.error('Only int8, int16 quantization  or no (float32) quantization are supported, got %s', quantize)
            raise ValueError

        self.__quantize = quantize
        self.__new_converter = new_converter

        if quantize == 'float32':
            self.__width = 32
        elif quantize in ('float16', 'int16'):
            self.__width = 16
        elif quantize in ('float8', 'int8'):
            self.__width = 8

    def __check_quantize(self, quantize: str) -> TypeGuard[QuantizeType]:
        return quantize in typing.get_args(QuantizeType)

    def representative_dataset_gen(self, representative_dataset: numpy.typing.NDArray[Any]) -> Generator[
            list[numpy.typing.NDArray[Any]], None, None]:
        for input_value in representative_dataset:
            yield [input_value.reshape([1, *input_value.shape])]

    @override
    def convert(self,
                framework: LearningFramework[keras.Model],
                model: keras.Model,
                model_name: str,
                representative_dataset: numpy.typing.NDArray[Any]) -> Keras2TFLite | None:
        # Another Keras 3 workaround
        if not hasattr(model, '_get_save_spec'):
            keras_function = tf.function(model.call,
                                         autograph=False,
                                         input_signature=[tf.TensorSpec(model.input_shape, tf.float32)])
            keras_concrete_function = keras_function.get_concrete_function(tf.TensorSpec(model.input_shape, tf.float32))
            converter = tf.lite.TFLiteConverter.from_concrete_functions([keras_concrete_function], keras_function)

        else:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)

        converter.experimental_new_converter = self.__new_converter

        if self.__quantize == 'float16':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]

            logger.error('float16 quantization is not supported')
            raise ValueError

        if self.__quantize == 'float8':
            converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]

            logger.error('float8 quantization is not supported')
            raise ValueError

        if self.__quantize == 'int8':
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

        self.__data = converter.convert()
        self.__input_shape = model.input_shape

        return self

    @property
    def data(self) -> bytes:
        return self.__data

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self.__input_shape

    @override
    def process_mem_params(self, mem_params: int) -> Callable[[LearningFramework[keras.Model], keras.Model], int]:
        def f(framework: LearningFramework[keras.Model], model: keras.Model) -> int:
            return (framework.n_params(model) * self.__width) // 8
        return f
