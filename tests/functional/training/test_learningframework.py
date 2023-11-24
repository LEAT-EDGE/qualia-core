from unittest import TestCase

class TestTrainingFrameworks(TestCase):
    def setUp(self):
        import numpy as np
        from qualia_core.datamodel import RawDataModel
        from qualia_core.datamodel.RawDataModel import RawData, RawDataSets
        train = RawData(np.array([[[1.0]], [[1.0]]], dtype=np.float32), np.array([[1, 0], [1, 0]]))
        test = RawData(np.array([[[1.0]], [[1.0]]], dtype=np.float32), np.array([[1, 0], [0, 1]]))
        self.__data = RawDataModel(RawDataSets(train=train, test=test), name='test_train')

    def test_train_keras(self):
        from qualia_core import qualia
        from qualia_core.learningframework import Keras
        from tensorflow.keras.layers import Input, Flatten, Dense, Activation
        from tensorflow.keras.models import Sequential

        model = lambda input_shape, *a, **kwa: Sequential((Input(input_shape), Flatten(), Dense(2), Activation('softmax')))

        framework = Keras()

        trainresult = qualia.train(self.__data,
                        train_epochs=1,
                        iteration=1,
                        model_name='test_train',
                        model=model,
                        optimizer={'kind': 'Adam'},
                        framework=framework,
                        )
        self.assertEqual(trainresult.name, 'test_train')
        self.assertEqual(trainresult.i, 1)
        self.assertEqual(trainresult.model.input_shape, (None, 1, 1))
        self.assertEqual(trainresult.model.output_shape, (None, 2))
        self.assertEqual(trainresult.mem_params, (2 + 2) * 4) # 2 weights, 2 biases, 4 bytes (float32)
        self.assertEqual(trainresult.acc, 0.5) # Same data in one or the other class, should have 50% acc
        self.assertEqual(trainresult.framework, framework)

    def test_train_pytorch(self):
        from qualia_core import qualia
        from qualia_core.learningframework import PyTorch
        import math
        from torch.nn import Sequential, Flatten, Linear, ReLU

        class Model(Sequential):
            def __init__(self, input_shape, output_shape):
                self.input_shape = input_shape
                self.output_shape = output_shape
                super().__init__(Flatten(), Linear(math.prod(input_shape), 2))

        framework = PyTorch(enable_progress_bar=False)

        trainresult = qualia.train(self.__data,
                        train_epochs=1,
                        iteration=1,
                        model_name='test_train',
                        model=Model,
                        optimizer={'kind': 'Adam'},
                        framework=framework,
                        )
        self.assertEqual(trainresult.name, 'test_train')
        self.assertEqual(trainresult.i, 1)
        #self.assertEqual(trainresult.model.input_shape, (None, 1, 1)) # Not supported in PyTorch
        #self.assertEqual(trainresult.model.output_shape, (None, 2)), Not supported in PyTorch
        self.assertEqual(trainresult.mem_params, (2 + 2) * 4) # 2 weights, 2 biases, 4 bytes (float32)
        self.assertEqual(trainresult.acc, 0.5) # Same data in one or the other class, should have 50% acc
        self.assertEqual(trainresult.framework, framework)
