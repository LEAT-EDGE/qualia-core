from unittest import TestCase
from pathlib import Path

class TestWSMNIST(TestCase):
    def setUp(self):
        self.path = Path('data')/'WSMNIST'

    def test_wsmnist_spoken(self):
        import numpy as np
        from qualia_core.dataset import WSMNIST

        dataset = WSMNIST(path=str(self.path), variant='spoken')()

        trainX = np.load(self.path/'data_sp_train.npy')
        trainY = np.load(self.path/'labels_train.npy')
        testX = np.load(self.path/'data_sp_test.npy')
        testY = np.load(self.path/'labels_test.npy')
        
        self.assertEqual(dataset.sets.train.x.shape, (60000, 39, 13))
        self.assertEqual(dataset.sets.test.x.shape, (10000, 39, 13))
        self.assertEqual(dataset.sets.train.y.shape, (60000, ))
        self.assertEqual(dataset.sets.test.y.shape, (10000, ))

        self.assertTrue(np.allclose(dataset.sets.train.x.reshape((60000, -1)), trainX))
        self.assertTrue(np.allclose(dataset.sets.test.x.reshape((10000, -1)), testX))
        self.assertTrue(np.allclose(dataset.sets.train.y, trainY))
        self.assertTrue(np.allclose(dataset.sets.test.y, testY))

    def test_wsmnist_written(self):
        import numpy as np
        from qualia_core.dataset import WSMNIST

        dataset = WSMNIST(path=str(self.path), variant='written')()

        trainX = np.load(self.path/'data_wr_train.npy')
        trainY = np.load(self.path/'labels_train.npy')
        testX = np.load(self.path/'data_wr_test.npy')
        testY = np.load(self.path/'labels_test.npy')
        
        self.assertEqual(dataset.sets.train.x.shape, (60000, 28, 28, 1))
        self.assertEqual(dataset.sets.test.x.shape, (10000, 28, 28, 1))
        self.assertEqual(dataset.sets.train.y.shape, (60000, ))
        self.assertEqual(dataset.sets.test.y.shape, (10000, ))

        self.assertTrue(np.allclose(dataset.sets.train.x.reshape((60000, -1)), trainX))
        self.assertTrue(np.allclose(dataset.sets.test.x.reshape((10000, -1)), testX))
        self.assertTrue(np.allclose(dataset.sets.train.y, trainY))
        self.assertTrue(np.allclose(dataset.sets.test.y, testY))
