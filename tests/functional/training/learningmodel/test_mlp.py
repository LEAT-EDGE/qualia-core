from typing import ClassVar
from qualia_core.datamodel.RawDataModel import RawDataModel


class TestModelsMLP:
    data: ClassVar[RawDataModel]

    @classmethod
    def setup_method(cls) -> None:
        import numpy as np
        from qualia_core.datamodel import RawDataModel
        from qualia_core.datamodel.RawDataModel import RawData, RawDataSets
        train = RawData(np.array([[[1.0]], [[1.0]]], dtype=np.float32), np.array([[1, 0], [1, 0]]))
        test = RawData(np.array([[[1.0]], [[1.0]]], dtype=np.float32), np.array([[1, 0], [0, 1]]))
        cls.data = RawDataModel(RawDataSets(train=train, test=test), name='test_mlp')

    def test_mlp_keras(self):
        from qualia_core import qualia
        from qualia_core.learningframework import Keras
        from qualia_core.learningmodel.keras import MLP
        from tensorflow.keras.layers import Dense, Activation, Flatten
        from tensorflow.keras.activations import relu, softmax

        model = MLP
        model_params = {'units': (10, 10)}

        framework = Keras()

        trainresult = qualia.train(TestModelsMLP.data,
                        train_epochs=1,
                        iteration=1,
                        model_name='test_mlp_keras',
                        model=model,
                        model_params=model_params,
                        optimizer={'kind': 'Adam'},
                        framework=framework,
                        )
        assert trainresult.name == 'test_mlp_keras'
        assert trainresult.i == 1
        assert trainresult.model.input_shape == (None, 1, 1)
        assert trainresult.model.output_shape == (None, 2)

        assert isinstance(trainresult.model.layers[0], Flatten)
        assert isinstance(trainresult.model.layers[1], Dense)
        assert trainresult.model.layers[1].kernel.shape == (1, 10)
        assert trainresult.model.layers[1].bias.shape == (10)
        assert isinstance(trainresult.model.layers[2], Activation)
        assert trainresult.model.layers[2].activation == relu
        assert isinstance(trainresult.model.layers[3], Dense)
        assert trainresult.model.layers[3].kernel.shape == (10, 10)
        assert trainresult.model.layers[3].bias.shape == (10)
        assert isinstance(trainresult.model.layers[4], Activation)
        assert trainresult.model.layers[4].activation == relu
        assert isinstance(trainresult.model.layers[5], Dense)
        assert trainresult.model.layers[5].kernel.shape == (10, 2)
        assert trainresult.model.layers[5].bias.shape == (2)
        assert isinstance(trainresult.model.layers[6], Activation)
        assert trainresult.model.layers[6].activation == softmax

        # first layer 10 weights/10 biases, second layer 10*10 weights (10 inputs, 10 outputs)/10 biases, 3rd layer 10*2 weights (10 inputs 2 outputs)/2 biases, 4 bytes (float32)
        assert trainresult.mem_params == (10 + 10 + 10*10 + 10 + 10*2 + 2) * 4
        assert trainresult.acc == 0.5 # Same data in one or the other class, should have 50% acc
        assert trainresult.framework == framework

    def test_mlp_pytorch(self) -> None:
        from qualia_core import qualia
        from qualia_core.learningframework import PyTorch
        from qualia_core.learningmodel.pytorch import MLP
        from torch.nn import Flatten, Linear, ReLU

        model = MLP
        model_params = {'units': (10, 10)}

        framework = PyTorch(enable_progress_bar=False)

        trainresult = qualia.train(TestModelsMLP.data,
                        train_epochs=1,
                        iteration=1,
                        model_name='test_mlp_pytorch',
                        model=model,
                        model_params=model_params,
                        optimizer={'kind': 'Adam'},
                        framework=framework,
                        )
        assert trainresult.name == 'test_mlp_pytorch'
        assert trainresult.i == 1
        #self.assert trainresult.model.input_shape, (None, 1, 1)) # Not supported in PyTorch
        #self.assert trainresult.model.output_shape, (None, 2)), Not supported in PyTorch
        # first layer 10 weights/10 biases, second layer 10*10 weights (10 inputs, 10 outputs)/10 biases, 3rd layer 10*2 weights (10 inputs 2 outputs)/2 biases, 4 bytes (float32)

        assert len(list(trainresult.model.children())) == 1
        assert len(list(trainresult.model.layers.children())) == 6
        assert isinstance(trainresult.model.layers.flatten1, Flatten)
        assert isinstance(trainresult.model.layers.fc1, Linear)
        assert trainresult.model.layers.fc1.weight.shape == (10, 1)
        assert trainresult.model.layers.fc1.bias.shape == (10, )
        assert isinstance(trainresult.model.layers.relu1, ReLU)
        assert isinstance(trainresult.model.layers.fc2, Linear)
        assert trainresult.model.layers.fc2.weight.shape == (10, 10)
        assert trainresult.model.layers.fc2.bias.shape == (10, )
        assert isinstance(trainresult.model.layers.relu2, ReLU)
        assert isinstance(trainresult.model.layers.fc3, Linear)
        assert trainresult.model.layers.fc3.weight.shape == (2, 10)
        assert trainresult.model.layers.fc3.bias.shape == (2, )

        assert trainresult.mem_params == (10 + 10 + 10*10 + 10 + 10*2 + 2) * 4
        assert trainresult.acc == 0.5 # Same data in one or the other class, should have 50% acc
        assert trainresult.framework, framework
