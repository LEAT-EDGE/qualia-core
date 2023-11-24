from unittest import TestCase

class TestModelsMLP(TestCase):
    def setUp(self):
        import numpy as np
        from qualia_core.datamodel import RawDataModel
        from qualia_core.datamodel.RawDataModel import RawData, RawDataSets
        train = RawData(np.array([[[1.0]], [[1.0]]], dtype=np.float32), np.array([[1, 0], [1, 0]]))
        test = RawData(np.array([[[1.0]], [[1.0]]], dtype=np.float32), np.array([[1, 0], [0, 1]]))
        self.__data = RawDataModel(RawDataSets(train=train, test=test), name='test_mlp')

    def test_mlp_keras(self):
        from qualia_core import qualia
        from qualia_core.learningframework import Keras
        from qualia_core.learningmodel.keras import MLP
        from tensorflow.keras.layers import Dense, Activation, Flatten
        from tensorflow.keras.activations import relu, softmax

        model = MLP
        model_params = {'units': (10, 10)}

        framework = Keras()

        trainresult = qualia.train(self.__data,
                        train_epochs=1,
                        iteration=1,
                        model_name='test_mlp_keras',
                        model=model,
                        model_params=model_params,
                        optimizer={'kind': 'Adam'},
                        framework=framework,
                        )
        self.assertEqual(trainresult.name, 'test_mlp_keras')
        self.assertEqual(trainresult.i, 1)
        self.assertEqual(trainresult.model.input_shape, (None, 1, 1))
        self.assertEqual(trainresult.model.output_shape, (None, 2))

        self.assertIsInstance(trainresult.model.layers[0], Flatten)
        self.assertIsInstance(trainresult.model.layers[1], Dense)
        self.assertEqual(trainresult.model.layers[1].kernel.shape, (1, 10))
        self.assertEqual(trainresult.model.layers[1].bias.shape, (10))
        self.assertIsInstance(trainresult.model.layers[2], Activation)
        self.assertEqual(trainresult.model.layers[2].activation, relu)
        self.assertIsInstance(trainresult.model.layers[3], Dense)
        self.assertEqual(trainresult.model.layers[3].kernel.shape, (10, 10))
        self.assertEqual(trainresult.model.layers[3].bias.shape, (10))
        self.assertIsInstance(trainresult.model.layers[4], Activation)
        self.assertEqual(trainresult.model.layers[4].activation, relu)
        self.assertIsInstance(trainresult.model.layers[5], Dense)
        self.assertEqual(trainresult.model.layers[5].kernel.shape, (10, 2))
        self.assertEqual(trainresult.model.layers[5].bias.shape, (2))
        self.assertIsInstance(trainresult.model.layers[6], Activation)
        self.assertEqual(trainresult.model.layers[6].activation, softmax)

        # first layer 10 weights/10 biases, second layer 10*10 weights (10 inputs, 10 outputs)/10 biases, 3rd layer 10*2 weights (10 inputs 2 outputs)/2 biases, 4 bytes (float32)
        self.assertEqual(trainresult.mem_params, (10 + 10 + 10*10 + 10 + 10*2 + 2) * 4)
        self.assertEqual(trainresult.acc, 0.5) # Same data in one or the other class, should have 50% acc
        self.assertEqual(trainresult.framework, framework)

    def test_mlp_pytorch(self):
        from qualia_core import qualia
        from qualia_core.learningframework import PyTorch
        from qualia_core.learningmodel.pytorch import MLP
        from torch.nn import Flatten, Linear, ReLU

        model = MLP
        model_params = {'units': (10, 10)}

        framework = PyTorch(enable_progress_bar=False)

        trainresult = qualia.train(self.__data,
                        train_epochs=1,
                        iteration=1,
                        model_name='test_mlp_pytorch',
                        model=model,
                        model_params=model_params,
                        optimizer={'kind': 'Adam'},
                        framework=framework,
                        )
        self.assertEqual(trainresult.name, 'test_mlp_pytorch')
        self.assertEqual(trainresult.i, 1)
        #self.assertEqual(trainresult.model.input_shape, (None, 1, 1)) # Not supported in PyTorch
        #self.assertEqual(trainresult.model.output_shape, (None, 2)), Not supported in PyTorch
        # first layer 10 weights/10 biases, second layer 10*10 weights (10 inputs, 10 outputs)/10 biases, 3rd layer 10*2 weights (10 inputs 2 outputs)/2 biases, 4 bytes (float32)

        self.assertIsInstance(trainresult.model[0], Flatten)
        self.assertIsInstance(trainresult.model[1], Linear)
        self.assertEqual(trainresult.model[1].weight.shape, (10, 1))
        self.assertEqual(tuple(trainresult.model[1].bias.shape), (10, ))
        self.assertIsInstance(trainresult.model[2], ReLU)
        self.assertIsInstance(trainresult.model[3], Linear)
        self.assertEqual(trainresult.model[3].weight.shape, (10, 10))
        self.assertEqual(trainresult.model[3].bias.shape, (10, ))
        self.assertIsInstance(trainresult.model[4], ReLU)
        self.assertIsInstance(trainresult.model[5], Linear)
        self.assertEqual(trainresult.model[5].weight.shape, (2, 10))
        self.assertEqual(trainresult.model[5].bias.shape, (2, ))

        self.assertEqual(trainresult.mem_params, (10 + 10 + 10*10 + 10 + 10*2 + 2) * 4)
        self.assertEqual(trainresult.acc, 0.5) # Same data in one or the other class, should have 50% acc
        self.assertEqual(trainresult.framework, framework)
