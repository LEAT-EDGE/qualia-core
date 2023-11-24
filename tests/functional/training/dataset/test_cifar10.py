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

        assert dataset.sets.train.x.shape == (50000, 32, 32, 3)
        assert dataset.sets.train.x.dtype == np.float32
        assert dataset.sets.test.x.shape == (10000, 32, 32, 3)
        assert dataset.sets.test.x.dtype == np.float32
        assert dataset.sets.train.y.shape == (50000, )
        assert dataset.sets.test.y.shape == (10000, )

        assert np.array_equal(dataset.sets.test.y, test_y)
