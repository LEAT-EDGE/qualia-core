from __future__ import annotations
import logging
import sys
from pathlib import Path
import numpy as np
from qualia_core.datamodel.RawDataModel import RawData, RawDataSets, RawDataModel
from qualia_core.dataset.RawDataset import RawDataset

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)


class MNISTBase(RawDataset):
    """Base class for MNIST-style datasets (MNIST and Fashion-MNIST).

    This class provides common functionality for loading and processing datasets that
    use the IDX file format. Both MNIST and Fashion-MNIST share the same:
    - File format (IDX)
    - Image dimensions (28x28 pixels)
    - Number of classes (10)
    - Dataset sizes (60,000 training, 10,000 test)

    The IDX file format is a simple format for vectors and multidimensional matrices
    of various numerical types. The files are organized as:
    - magic number (4 bytes) identifying data type and dimensions
    - dimension sizes (4 bytes each)
    - data in row-major order
    """

    def __init__(self, path: str = '', variant: str = 'raw') -> None:
        """Initialize an MNIST-style dataset.

        Args:
            path: Directory containing the IDX files
            variant: Dataset variant (default: 'raw'). Currently unused but
                    maintained for consistency with other Qualia datasets.
        """
        super().__init__()
        self.__path = Path(path)
        self.__variant = variant

        # MNIST datasets don't use a validation set, so we remove it
        if 'valid' in self.sets:
            self.sets.remove('valid')

    def _read_idx_file(self, filepath: Path) -> np.ndarray:
        """Read data from an IDX file format.

        The IDX file format begins with a magic number containing:
        - first 2 bytes: zero
        - third byte: data type (0x08: unsigned byte)
        - fourth byte: number of dimensions
        Following this are the dimension sizes (4 bytes each)
        Finally comes the data in row-major order.

        Args:
            filepath: Path to the IDX file to read

        Returns:
            numpy.ndarray containing the file's data properly shaped

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is invalid
        """
        if not filepath.exists():
            raise FileNotFoundError(f"IDX file not found: {filepath}")

        with filepath.open('rb') as f:
            # Read the magic number
            magic = int.from_bytes(f.read(4), byteorder='big')
            n_dims = magic % 256  # Last byte is number of dimensions

            # Verify this is an unsigned byte IDX file (type 0x08)
            dtype = (magic >> 8) % 256
            if dtype != 0x08:
                raise ValueError(f"Expected unsigned byte data (0x08), got {dtype}")

            # Read the dimension sizes
            dims = []
            for _ in range(n_dims):
                dims.append(int.from_bytes(f.read(4), byteorder='big'))

            # Read all the data and reshape to the specified dimensions
            data = np.frombuffer(f.read(), dtype=np.uint8)
            data = data.reshape(dims)

            return data

    def _load_data(self, images_file: str, labels_file: str) -> tuple[np.ndarray, np.ndarray]:
        """Load and preprocess a set of images and labels.

        This method:
        1. Reads both the image and label IDX files
        2. Reshapes images to [N, H, W, C] format as required by Qualia
        3. Normalizes pixel values to [0, 1] range
        4. Ensures data types are correct (float32 for images)

        Args:
            images_file: Name of the IDX file containing images
            labels_file: Name of the IDX file containing labels

        Returns:
            Tuple of (images, labels) where:
            - images is float32 array of shape [N, 28, 28, 1], values in [0, 1]
            - labels is uint8 array of shape [N] with values 0-9
        """
        # Load raw data from IDX files
        images = self._read_idx_file(self.__path / images_file)
        labels = self._read_idx_file(self.__path / labels_file)

        # Images need to be:
        # - Reshaped to [N, H, W, C] format (adding channel dimension)
        # - Converted to float32 for training
        # - Normalized to [0, 1] range
        images = images.reshape(-1, 28, 28, 1).astype(np.float32) / 255.0

        return images, labels

    @override
    def __call__(self) -> RawDataModel:
        """Load and prepare the complete dataset.

        This method:
        1. Loads both training and test sets
        2. Formats them according to Qualia's requirements
        3. Packages them in Qualia's data structures

        The MNIST datasets use specific file names:
        - train-images-idx3-ubyte: Training images (60,000 x 28 x 28)
        - train-labels-idx1-ubyte: Training labels (60,000)
        - t10k-images-idx3-ubyte: Test images (10,000 x 28 x 28)
        - t10k-labels-idx1-ubyte: Test labels (10,000)

        Returns:
            RawDataModel containing:
            - Training set (60,000 samples)
            - Test set (10,000 samples)
            Each set has:
            - Images: float32 [N, 28, 28, 1] arrays, values in [0, 1]
            - Labels: uint8 [N] arrays with values 0-9
        """
        logger.info('Loading MNIST-style dataset from %s', self.__path)

        # Load training and test sets
        train_x, train_y = self._load_data('train-images-idx3-ubyte',
                                           'train-labels-idx1-ubyte')
        test_x, test_y = self._load_data('t10k-images-idx3-ubyte',
                                         't10k-labels-idx1-ubyte')

        # Log shapes to verify loading was correct
        logger.info('Shapes: train_x=%s, train_y=%s, test_x=%s, test_y=%s',
                    train_x.shape, train_y.shape, test_x.shape, test_y.shape)

        # Package everything in Qualia's containers
        return RawDataModel(
            sets=RawDataSets(
                train=RawData(train_x, train_y),
                test=RawData(test_x, test_y)
            ),
            name=self.name
        )

    @property
    @override
    def name(self) -> str:
        """Get the dataset name.

        Returns a combination of the class name and variant,
        used for logging and file naming.

        Returns:
            String like 'MNIST_raw' or 'FashionMNIST_raw'
        """
        return f'{self.__class__.__name__}_{self.__variant}'


class MNIST(MNISTBase):
    """Original MNIST handwritten digits dataset.

    The MNIST database contains 70,000 grayscale images of handwritten digits (0-9).
    Each image is 28x28 pixels, centered to reduce preprocessing and get better results.

    Dataset split:
    - 60,000 training images
    - 10,000 test images

    Labels:
    - 0-9: Corresponding digits
    """
    pass


class FashionMNIST(MNISTBase):
    """Fashion MNIST clothing dataset.

    A drop-in replacement for MNIST, containing 70,000 grayscale images of clothing items.
    Each image is 28x28 pixels, following the same format as original MNIST.

    Dataset split:
    - 60,000 training images
    - 10,000 test images

    Labels:
    0: T-shirt/top    5: Sandal
    1: Trouser        6: Shirt
    2: Pullover       7: Sneaker
    3: Dress          8: Bag
    4: Coat           9: Ankle boot
    """
    pass