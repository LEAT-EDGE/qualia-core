from pathlib import Path


class TestCIFAR10:
    path = Path('data')/'cifar-10-batches-py'

    def test_cifar10(self) -> None:
        import numpy as np
        from qualia_core.dataset import CIFAR10

        dataset = CIFAR10(path=str(self.path))()

        import pickle
        with (self.path/'test_batch').open('rb') as fo:
            raw = pickle.load(fo, encoding='bytes')
            test_y = np.array(raw[b'labels'])

        train_batches = [chunk for chunk in dataset.sets.train.chunks]
        train_x = np.concatenate([b.data for b in train_batches])
        train_y = np.concatenate([b.labels for b in train_batches])
        del train_batches
        assert train_x.shape == (50000, 32, 32, 3)
        assert train_x.dtype == np.float32
        del train_x
        assert train_y.shape == (50000, )
        del train_y

        test_batches = [chunk for chunk in dataset.sets.test.chunks]
        test_x = np.concatenate([b.data for b in test_batches])
        test_y = np.concatenate([b.labels for b in test_batches])
        del test_batches
        assert test_x.shape == (10000, 32, 32, 3)
        assert test_x.dtype == np.float32
        del test_x
        assert test_y.shape == (10000, )

        assert np.array_equal(test_y, test_y)

        del test_y
