from unittest import TestCase

class TestModelsCNN1D(TestCase):
    def setUp(self):
        import numpy as np
        from qualia_core.datamodel import RawDataModel
        from qualia_core.datamodel.RawDataModel import RawData, RawDataSets
        train = RawData(np.ones((2, 64, 1), dtype=np.float32), np.array([[1, 0], [1, 0]]))
        test = RawData(np.ones((2, 64, 1), dtype=np.float32), np.array([[1, 0], [0, 1]]))
        self.__data = RawDataModel(RawDataSets(train=train, test=test), name='test_cnn1d')
        self.__model_params = {
            'filters': (4, 6),
            'fc_units': (10, 10),
            'kernel_sizes': (5, 3),
            'batch_norm': True,
            'dropouts': [0.5, 0.5, 0, 0],
            'pool_sizes': (4, 2),
            'prepool': 2,
            'dims': 1
            }

    def test_cnn_1d_keras(self):
        from qualia_core import qualia
        from qualia_core.learningframework import Keras
        from qualia_core.learningmodel.keras import CNN
        from tensorflow.keras.layers import Dense, Activation, Flatten, Conv1D, AveragePooling1D, MaxPooling1D, BatchNormalization, Dropout
        from tensorflow.keras.activations import relu, softmax

        model = CNN

        framework = Keras()

        trainresult = qualia.train(self.__data,
                        train_epochs=1,
                        iteration=1,
                        model_name='test_cnn_1d_keras',
                        model=model,
                        model_params=self.__model_params,
                        optimizer={'kind': 'Adam'},
                        framework=framework,
                        )
        self.assertEqual(trainresult.name, 'test_cnn_1d_keras')
        self.assertEqual(trainresult.i, 1)
        self.assertEqual(trainresult.model.input_shape, (None, 64, 1))
        self.assertEqual(trainresult.model.output_shape, (None, 2))


        # prepool
        self.assertIsInstance(trainresult.model.layers[0], AveragePooling1D)
        self.assertEqual(tuple(trainresult.model.layers[0].pool_size), (2,))
        self.assertEqual(tuple(trainresult.model.layers[0].strides), (2,))

        # 1st conv block
        self.assertIsInstance(trainresult.model.layers[1], Conv1D)
        self.assertEqual(tuple(trainresult.model.layers[1].kernel.shape), (5, 1, 4)) # 4*1*5 = 20
        self.assertEqual(tuple(trainresult.model.layers[1].bias.shape), (4, )) # 4
        self.assertIsInstance(trainresult.model.layers[2], Activation)
        self.assertEqual(trainresult.model.layers[2].activation, relu)
        self.assertIsInstance(trainresult.model.layers[3], BatchNormalization)
        self.assertEqual(tuple(trainresult.model.layers[3].moving_mean.shape), (4,)) # 4
        self.assertEqual(tuple(trainresult.model.layers[3].moving_variance.shape), (4, )) # 4
        self.assertEqual(tuple(trainresult.model.layers[3].gamma.shape), (4, )) # 4
        self.assertEqual(tuple(trainresult.model.layers[3].beta.shape), (4, )) # 4
        self.assertIsInstance(trainresult.model.layers[4], Dropout)
        self.assertEqual(trainresult.model.layers[4].rate, 0.5)
        self.assertIsInstance(trainresult.model.layers[5], MaxPooling1D)
        self.assertEqual(tuple(trainresult.model.layers[5].pool_size), (4,))
        self.assertEqual(tuple(trainresult.model.layers[5].strides), (4,))

        # 2nd conv block
        self.assertIsInstance(trainresult.model.layers[6], Conv1D)
        self.assertEqual(tuple(trainresult.model.layers[6].kernel.shape), (3, 4, 6)) # 6*4*3 = 72
        self.assertEqual(tuple(trainresult.model.layers[6].bias.shape), (6, )) # 6
        self.assertIsInstance(trainresult.model.layers[7], Activation)
        self.assertEqual(trainresult.model.layers[7].activation, relu)
        self.assertIsInstance(trainresult.model.layers[8], BatchNormalization)
        self.assertEqual(tuple(trainresult.model.layers[8].moving_mean.shape), (6,)) # 6
        self.assertEqual(tuple(trainresult.model.layers[8].moving_variance.shape), (6, )) # 6
        self.assertEqual(tuple(trainresult.model.layers[8].gamma.shape), (6, )) # 6
        self.assertEqual(tuple(trainresult.model.layers[8].beta.shape), (6, )) # 6
        self.assertIsInstance(trainresult.model.layers[9], Dropout)
        self.assertEqual(trainresult.model.layers[9].rate, 0.5)
        self.assertIsInstance(trainresult.model.layers[10], MaxPooling1D)
        self.assertEqual(tuple(trainresult.model.layers[10].pool_size), (2,))
        self.assertEqual(tuple(trainresult.model.layers[10].strides), (2,))

        self.assertIsInstance(trainresult.model.layers[11], Flatten)

        # fc
        self.assertIsInstance(trainresult.model.layers[12], Dense)
        self.assertEqual(trainresult.model.layers[12].kernel.shape, (12, 10)) # 120
        self.assertEqual(trainresult.model.layers[12].bias.shape, (10, )) # 10
        self.assertIsInstance(trainresult.model.layers[13], Activation)
        self.assertEqual(trainresult.model.layers[13].activation, relu)
        self.assertIsInstance(trainresult.model.layers[14], Dense)
        self.assertEqual(trainresult.model.layers[14].kernel.shape, (10, 10)) # 100
        self.assertEqual(trainresult.model.layers[14].bias.shape, (10, )) # 10
        self.assertIsInstance(trainresult.model.layers[15], Activation)
        self.assertEqual(trainresult.model.layers[15].activation, relu)
        self.assertIsInstance(trainresult.model.layers[16], Dense)
        self.assertEqual(trainresult.model.layers[16].kernel.shape, (10, 2)) # 20
        self.assertEqual(trainresult.model.layers[16].bias.shape, (2, )) # 2

        # softmax
        self.assertIsInstance(trainresult.model.layers[17], Activation)
        self.assertEqual(trainresult.model.layers[17].activation, softmax)


        # first layer 10 weights/10 biases, second layer 10*10 weights (10 inputs, 10 outputs)/10 biases, 3rd layer 10*2 weights (10 inputs 2 outputs)/2 biases, 4 bytes (float32)
        self.assertEqual(trainresult.mem_params, (4*1*5 + 4 + 4*4 + 6*4*3 + 6 + 6*4 + 10*12 +10 + 10*10 + 10 + 2*10 + 2) * 4)
        self.assertEqual(trainresult.acc, 0.5) # Same data in one or the other class, should have 50% acc
        self.assertEqual(trainresult.framework, framework)

    def test_cnn_1d_pytorch(self):
        from qualia_core import qualia
        from qualia_core.learningframework import PyTorch
        from qualia_core.learningmodel.pytorch import CNN
        from torch.nn import Flatten, Linear, ReLU, AvgPool1d, MaxPool1d, Conv1d, BatchNorm1d, Dropout

        model = CNN

        framework = PyTorch(enable_progress_bar=False)

        trainresult = qualia.train(self.__data,
                        train_epochs=1,
                        iteration=1,
                        model_name='test_cnn_1d_pytorch',
                        model=model,
                        model_params=self.__model_params,
                        optimizer={'kind': 'Adam'},
                        framework=framework,
                        )
        self.assertEqual(trainresult.name, 'test_cnn_1d_pytorch')
        self.assertEqual(trainresult.i, 1)
        #self.assertEqual(trainresult.model.input_shape, (None, 1, 1)) # Not supported in PyTorch
        #self.assertEqual(trainresult.model.output_shape, (None, 2)), Not supported in PyTorch


        # prepool
        self.assertIsInstance(trainresult.model[0], AvgPool1d)
        self.assertEqual(tuple(trainresult.model[0].kernel_size), (2,))
        self.assertEqual(tuple(trainresult.model[0].stride), (2,))

        # 1st conv block
        self.assertIsInstance(trainresult.model[1], Conv1d)
        self.assertEqual(tuple(trainresult.model[1].weight.shape), (4, 1, 5)) # 4*1*5 = 20
        self.assertEqual(tuple(trainresult.model[1].bias.shape), (4, )) # 4
        self.assertIsInstance(trainresult.model[2], BatchNorm1d)
        self.assertEqual(tuple(trainresult.model[2].weight.shape), (4,)) # 4
        self.assertEqual(tuple(trainresult.model[2].bias.shape), (4, )) # 4
        self.assertIsInstance(trainresult.model[3], ReLU)
        self.assertIsInstance(trainresult.model[4], Dropout)
        self.assertEqual(trainresult.model[4].p, 0.5)
        self.assertIsInstance(trainresult.model[5], MaxPool1d)
        self.assertEqual(trainresult.model[5].kernel_size, 4)
        self.assertEqual(trainresult.model[5].stride, 4)

        # 2nd conv block
        self.assertIsInstance(trainresult.model[6], Conv1d)
        self.assertEqual(tuple(trainresult.model[6].weight.shape), (6, 4, 3)) # 6*4*3 = 72
        self.assertEqual(tuple(trainresult.model[6].bias.shape), (6, )) # 6
        self.assertIsInstance(trainresult.model[7], BatchNorm1d)
        self.assertEqual(tuple(trainresult.model[7].weight.shape), (6,)) # 6
        self.assertEqual(tuple(trainresult.model[7].bias.shape), (6, )) # 6
        self.assertIsInstance(trainresult.model[8], ReLU)
        self.assertIsInstance(trainresult.model[9], Dropout)
        self.assertEqual(trainresult.model[9].p, 0.5)
        self.assertIsInstance(trainresult.model[10], MaxPool1d)
        self.assertEqual(trainresult.model[10].kernel_size, 2)
        self.assertEqual(trainresult.model[10].stride, 2)

        self.assertIsInstance(trainresult.model[11], Flatten)

        # fc
        self.assertIsInstance(trainresult.model[12], Linear)
        self.assertEqual(trainresult.model[12].weight.shape, (10, 12)) # 120
        self.assertEqual(trainresult.model[12].bias.shape, (10, )) # 10
        self.assertIsInstance(trainresult.model[13], ReLU)
        self.assertIsInstance(trainresult.model[14], Linear)
        self.assertEqual(trainresult.model[14].weight.shape, (10, 10)) # 100
        self.assertEqual(trainresult.model[14].bias.shape, (10, )) # 10
        self.assertIsInstance(trainresult.model[15], ReLU)
        self.assertIsInstance(trainresult.model[16], Linear)
        self.assertEqual(trainresult.model[16].weight.shape, (2, 10)) # 20
        self.assertEqual(trainresult.model[16].bias.shape, (2, )) # 2


        self.assertEqual(trainresult.mem_params, (4*1*5 + 4 + 4*2 + 6*4*3 + 6 + 6*2 + 10*12 +10 + 10*10 + 10 + 2*10 + 2) * 4)
        self.assertEqual(trainresult.acc, 0.5) # Same data in one or the other class, should have 50% acc
        self.assertEqual(trainresult.framework, framework)
