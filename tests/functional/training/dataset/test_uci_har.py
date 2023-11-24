from unittest import TestCase
from pathlib import Path

class TestUCI_HAR(TestCase):
    def setUp(self):
        self.path = Path('data')/'UCI HAR Dataset'

    def test_uci_har_raw(self):
        import numpy as np
        from qualia_core.dataset import UCI_HAR

        dataset = UCI_HAR(path=str(self.path), variant='raw')()

        train_data = np.concatenate([np.array(a.get_raw_array()) for s in dataset.sets.train for a in s.activities])
        test_data = np.concatenate([np.array(a.get_raw_array()) for s in dataset.sets.test for a in s.activities])
        train_labels = np.concatenate([np.full(len(a.samples), a.kind) for s in dataset.sets.train for a in s.activities])
        test_labels = np.concatenate([np.full(len(a.samples), a.kind) for s in dataset.sets.test for a in s.activities])

        train_bodyaccel_x = np.loadtxt(self.path/'train'/'Inertial Signals'/'body_acc_x_train.txt')
        train_bodyaccel_y = np.loadtxt(self.path/'train'/'Inertial Signals'/'body_acc_y_train.txt')
        train_bodyaccel_z = np.loadtxt(self.path/'train'/'Inertial Signals'/'body_acc_z_train.txt')
        train_bodygyro_x = np.loadtxt(self.path/'train'/'Inertial Signals'/'body_gyro_x_train.txt')
        train_bodygyro_y = np.loadtxt(self.path/'train'/'Inertial Signals'/'body_gyro_y_train.txt')
        train_bodygyro_z = np.loadtxt(self.path/'train'/'Inertial Signals'/'body_gyro_z_train.txt')
        train_totalaccel_x = np.loadtxt(self.path/'train'/'Inertial Signals'/'total_acc_x_train.txt')
        train_totalaccel_y = np.loadtxt(self.path/'train'/'Inertial Signals'/'total_acc_y_train.txt')
        train_totalaccel_z = np.loadtxt(self.path/'train'/'Inertial Signals'/'total_acc_z_train.txt')

        test_bodyaccel_x = np.loadtxt(self.path/'test'/'Inertial Signals'/'body_acc_x_test.txt')
        test_bodyaccel_y = np.loadtxt(self.path/'test'/'Inertial Signals'/'body_acc_y_test.txt')
        test_bodyaccel_z = np.loadtxt(self.path/'test'/'Inertial Signals'/'body_acc_z_test.txt')
        test_bodygyro_x = np.loadtxt(self.path/'test'/'Inertial Signals'/'body_gyro_x_test.txt')
        test_bodygyro_y = np.loadtxt(self.path/'test'/'Inertial Signals'/'body_gyro_y_test.txt')
        test_bodygyro_z = np.loadtxt(self.path/'test'/'Inertial Signals'/'body_gyro_z_test.txt')
        test_totalaccel_x = np.loadtxt(self.path/'test'/'Inertial Signals'/'total_acc_x_test.txt')
        test_totalaccel_y = np.loadtxt(self.path/'test'/'Inertial Signals'/'total_acc_y_test.txt')
        test_totalaccel_z = np.loadtxt(self.path/'test'/'Inertial Signals'/'total_acc_z_test.txt')
            
        train_y = np.loadtxt(self.path/'train'/'y_train.txt')
        test_y = np.loadtxt(self.path/'test'/'y_test.txt')

        self.assertEqual(dataset.name, 'UCI_HAR_raw')
        self.assertEqual(len(dataset.sets.train), 21)
        self.assertEqual(len(dataset.sets.test), 9)
        
        self.assertEqual(train_data.shape, (7352, 128, 9))
        self.assertEqual(test_data.shape, (2947, 128, 9))

        self.assertTrue(np.allclose(test_data[:,:,0], test_bodyaccel_x))
        self.assertTrue(np.allclose(test_data[:,:,1], test_bodyaccel_y))
        self.assertTrue(np.allclose(test_data[:,:,2], test_bodyaccel_z))
        self.assertTrue(np.allclose(test_data[:,:,3], test_bodygyro_x))
        self.assertTrue(np.allclose(test_data[:,:,4], test_bodygyro_y))
        self.assertTrue(np.allclose(test_data[:,:,5], test_bodygyro_z))
        self.assertTrue(np.allclose(test_data[:,:,6], test_totalaccel_x))
        self.assertTrue(np.allclose(test_data[:,:,7], test_totalaccel_y))
        self.assertTrue(np.allclose(test_data[:,:,8], test_totalaccel_z))

        self.assertTrue(np.allclose(train_data[:,:,0], train_bodyaccel_x))
        self.assertTrue(np.allclose(train_data[:,:,1], train_bodyaccel_y))
        self.assertTrue(np.allclose(train_data[:,:,2], train_bodyaccel_z))
        self.assertTrue(np.allclose(train_data[:,:,3], train_bodygyro_x))
        self.assertTrue(np.allclose(train_data[:,:,4], train_bodygyro_y))
        self.assertTrue(np.allclose(train_data[:,:,5], train_bodygyro_z))
        self.assertTrue(np.allclose(train_data[:,:,6], train_totalaccel_x))
        self.assertTrue(np.allclose(train_data[:,:,7], train_totalaccel_y))
        self.assertTrue(np.allclose(train_data[:,:,8], train_totalaccel_z))

        self.assertTrue(np.array_equal(train_labels, train_y-1)) # Original file is indexed from 1
