from unittest import TestCase
from pathlib import Path

class TestGTSRB(TestCase):
    def setUp(self):
        self.path = Path('data')/'GTSRB'

    def test_gtsrb(self):
        import numpy as np
        from qualia_core.dataset import GTSRB

        dataset = GTSRB(path=str(self.path), width=8, height=8)()

        trainY = [int(f.parent.name) for f in (self.path/'Final_Training'/'Images').glob('*/*.ppm')]

        with open(self.path/'GT-final_test.csv') as f:
            next(f) # Skip first line
            testY = np.array([int(l.split(';')[-1]) for l in f.read().splitlines()])

        self.assertEqual(dataset.sets.train.x.shape, (39209, 8, 8, 3))
        self.assertEqual(dataset.sets.test.x.shape, (12630, 8, 8, 3))
        self.assertEqual(dataset.sets.train.y.shape, (39209, ))
        self.assertEqual(dataset.sets.test.y.shape, (12630, ))

        self.assertTrue(np.array_equal(dataset.sets.train.y, trainY))
        self.assertTrue(np.array_equal(dataset.sets.test.y, testY))
