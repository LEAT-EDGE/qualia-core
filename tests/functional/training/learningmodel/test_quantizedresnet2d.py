from unittest import TestCase

class TestModelsQuantizedResnet2D(TestCase):
    def setUp(self):
        import numpy as np
        from qualia_core.datamodel import RawDataModel
        from qualia_core.datamodel.RawDataModel import RawData, RawDataSets
        train = RawData(np.ones((2, 64, 64, 1), dtype=np.float32), np.array([[1, 0], [1, 0]]))
        test = RawData(np.ones((2, 64, 64, 1), dtype=np.float32), np.array([[1, 0], [0, 1]]))
        self.__data = RawDataModel(RawDataSets(train=train, test=test), name='test_quantizedresnet2d')
        self.__model_params = {
            'filters': (4, 6),
            'kernel_sizes': (5, 5),

            'num_blocks': (2,),
            'strides': (1, 2),
            'paddings': (2, 2),

            'prepool': 1,
            'batch_norm': False,
            'bn_momentum': 0.95,

            'dims': 2,

            'quant_params': {'bits': 8,
                             'quantype': 'fxp',
                             'roundtype': 'floor',
                             'range_setting': 'minmax',
                             'LSQ': False},
            'fused_relu': True,
            }

    def test_quantized_resnet_2d_pytorch(self):
        from qualia_core import qualia
        from qualia_core.learningframework import PyTorch
        from qualia_core.learningmodel.pytorch import QuantizedResNet
        from qualia_core.learningmodel.pytorch.layers import QuantizedAdd
        from qualia_core.learningmodel.pytorch.quantized_layers2d import QuantizedConv2d, QuantizedMaxPool2d
        from qualia_core.learningmodel.pytorch.layers.quantized_layers import QuantizedIdentity, QuantizedLinear
        from torch.nn import Flatten, Linear, ReLU

        model = QuantizedResNet

        framework = PyTorch(enable_progress_bar=False)

        trainresult = qualia.train(self.__data,
                        train_epochs=1,
                        iteration=1,
                        model_name='test_quantized_resnet_2d_pytorch',
                        model=model,
                        model_params=self.__model_params,
                        optimizer={'kind': 'Adam'},
                        framework=framework,
                        )
        self.assertEqual(trainresult.name, 'test_quantized_resnet_2d_pytorch')
        self.assertEqual(trainresult.i, 1)
        #self.assertEqual(trainresult.model.input_shape, (None, 1, 1)) # Not supported in PyTorch
        #self.assertEqual(trainresult.model.output_shape, (None, 2)), Not supported in PyTorch

        self.assertEqual(len(list(trainresult.model.children())), 6)
        self.assertEqual(len(list(trainresult.model.layers.children())), 1)
        self.assertEqual(len(list(trainresult.model.layers[0].children())), 2)
        self.assertEqual(len(list(trainresult.model.layers[0][0].children())), 6)
        self.assertEqual(len(list(trainresult.model.layers[0][1].children())), 3)

        # input quantization
        self.assertEqual(type(trainresult.model.identity1), QuantizedIdentity)

        # prepool
        self.assertFalse(hasattr(trainresult.model, 'prepool'))

        # preconv
        self.assertEqual(type(trainresult.model.conv1), QuantizedConv2d)
        self.assertEqual(tuple(trainresult.model.conv1.padding), (2, 2))
        self.assertEqual(tuple(trainresult.model.conv1.weight.shape), (4, 1, 5, 5)) # 4*1*5*5 = 100
        self.assertEqual(tuple(trainresult.model.conv1.bias.shape), (4, )) # 4
        self.assertFalse(hasattr(trainresult.model, 'bn1'))
        self.assertEqual(type(trainresult.model.conv1.activation), ReLU)

        # 1st res block
        # 1st conv
        self.assertEqual(type(trainresult.model.layers[0][0].conv1), QuantizedConv2d)
        self.assertEqual(tuple(trainresult.model.layers[0][0].conv1.padding), (2, 2))
        self.assertEqual(tuple(trainresult.model.layers[0][0].conv1.weight.shape), (6, 4, 5, 5)) # 6*4*5*5 = 600
        self.assertEqual(tuple(trainresult.model.layers[0][0].conv1.bias.shape), (6, )) # 6
        self.assertFalse(hasattr(trainresult.model.layers[0][0], 'bn1'))
        self.assertEqual(type(trainresult.model.layers[0][0].pool1), QuantizedMaxPool2d)
        self.assertEqual(trainresult.model.layers[0][0].pool1.kernel_size, 2)
        self.assertEqual(trainresult.model.layers[0][0].pool1.stride, 2)
        self.assertEqual(type(trainresult.model.layers[0][0].pool1.activation), ReLU)
        # 2nd conv
        self.assertEqual(type(trainresult.model.layers[0][0].conv2), QuantizedConv2d)
        self.assertEqual(tuple(trainresult.model.layers[0][0].conv2.padding), (2, 2))
        self.assertEqual(tuple(trainresult.model.layers[0][0].conv2.weight.shape), (6, 6, 5, 5)) # 6*6*5*5 = 900
        self.assertEqual(tuple(trainresult.model.layers[0][0].conv2.bias.shape), (6, )) # 6
        self.assertFalse(hasattr(trainresult.model.layers[0][0], 'bn2'))
        # shortcut
        self.assertEqual(type(trainresult.model.layers[0][0].sconv), QuantizedConv2d)
        self.assertEqual(tuple(trainresult.model.layers[0][0].sconv.padding), (0, 0))
        self.assertEqual(tuple(trainresult.model.layers[0][0].sconv.weight.shape), (6, 4, 1, 1)) # 6*4*1*1 = 24
        self.assertEqual(tuple(trainresult.model.layers[0][0].sconv.bias.shape), (6, )) # 6
        self.assertFalse(hasattr(trainresult.model.layers[0][0], 'sbn'))
        self.assertEqual(type(trainresult.model.layers[0][0].smax), QuantizedMaxPool2d)
        self.assertEqual(trainresult.model.layers[0][0].smax.kernel_size, 2)
        self.assertEqual(trainresult.model.layers[0][0].smax.stride, 2)
        # add
        self.assertEqual(type(trainresult.model.layers[0][0].add), QuantizedAdd)
        self.assertEqual(type(trainresult.model.layers[0][0].add.activation), ReLU)


        # 2nd conv block
        # 1st conv
        self.assertEqual(type(trainresult.model.layers[0][1].conv1), QuantizedConv2d)
        self.assertEqual(tuple(trainresult.model.layers[0][1].conv1.padding), (2, 2))
        self.assertEqual(tuple(trainresult.model.layers[0][1].conv1.weight.shape), (6, 6, 5, 5)) # 6*6*5*5 = 900
        self.assertEqual(tuple(trainresult.model.layers[0][1].conv1.bias.shape), (6, )) # 6
        self.assertFalse(hasattr(trainresult.model.layers[0][1], 'bn1'))
        self.assertEqual(type(trainresult.model.layers[0][1].conv1.activation), ReLU)
        # 2nd conv
        self.assertEqual(type(trainresult.model.layers[0][1].conv2), QuantizedConv2d)
        self.assertEqual(tuple(trainresult.model.layers[0][1].conv2.padding), (2, 2))
        self.assertEqual(tuple(trainresult.model.layers[0][1].conv2.weight.shape), (6, 6, 5, 5)) # 6*6*5*5 = 900
        self.assertEqual(tuple(trainresult.model.layers[0][1].conv2.bias.shape), (6, )) # 6
        self.assertFalse(hasattr(trainresult.model.layers[0][1], 'bn2'))
        # add
        self.assertEqual(type(trainresult.model.layers[0][1].add), QuantizedAdd)
        self.assertEqual(type(trainresult.model.layers[0][1].add.activation), ReLU)

        self.assertEqual(type(trainresult.model.postpool), QuantizedMaxPool2d)
        self.assertEqual(tuple(trainresult.model.postpool.kernel_size), (32, 32))
        self.assertEqual(tuple(trainresult.model.postpool.stride), (32, 32))

        self.assertEqual(type(trainresult.model.flatten), Flatten)

        # fc
        self.assertEqual(type(trainresult.model.linear), QuantizedLinear)
        self.assertEqual(trainresult.model.linear.weight.shape, (2, 6)) # 2*6 = 12
        self.assertEqual(trainresult.model.linear.bias.shape, (2, )) # 2


        self.assertEqual(trainresult.mem_params, (5*5*1*4 + 4 + 5*5*4*6 + 6 + 5*5*6*6 + 6 + 1*1*4*6 + 6 + 5*5*6*6 + 6 + 5*5*6*6 + 6 + 6*2 + 2) * 4)
        self.assertEqual(trainresult.acc, 0.5) # Same data in one or the other class, should have 50% acc
        self.assertEqual(trainresult.framework, framework)
