"""##  EuroSAT.py: A Qualia dataset for land use and land cover classification using Sentinel-2 satellite images.

## Author
# - **Jonathan Courtois**
#   [jonathan.courtois@univ-cotedazur.fr](mailto:jonathan.courtois@univ-cotedazur.fr)
## Dataset Reference
# - **EuroSat Dataset:**
#   https://github.com/phelber/EuroSAT
# - **Installation Instructions:**
#   The EuroSat dataset .zip files must be uncompressed in your 'dataset' folder of the Qualia repository with this structure:
#   .dataset/
#   ├── EuroSAT/
#   ├── ├── MS/
#   ├── ├── ├── [Class_folder]/
#   ├── ├── ├── ├── [*.tif]
#   ├── ├── RGB/
#   ├── ├── ├── [Class_folder]/
#   ├── ├── ├── ├── [*.jpg]
# - **Citation:**
# @article{helber2019eurosat,
#     title={Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification},
#     author={Helber, Patrick and Bischke, Benjamin and Dengel, Andreas and Borth, Damian},
#     journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
#     year={2019},
#     publisher={IEEE}
"""

from __future__ import annotations  # annotations: Enables using class names in type hints before they're defined

import logging  # logging: For keeping track of what our dataset is doing
import time
from genericpath import exists
from pathlib import Path  # Path: Makes file handling consistent across operating systems

import numpy as np  # numpy: For efficient array operations on our data
import numpy.typing as npt  # npt: For type hints on numpy arrays, making our code clearer
import tifffile as tiff
from PIL import Image  # Image: For handling image files, especially for RGB images

from qualia_core import random  # Generator: For generating random numbers, useful for splitting data into training and test sets
from qualia_core.datamodel.RawDataModel import (  # RawData, RawDataSets, RawDataModel: The containers that Qualia expects
    RawData,
    RawDataModel,
    RawDataSets,
)
from qualia_core.dataset.RawDataset import (
    RawDataset,  # RawDataset: The base class that tells Qualia how to interact with our dataset
)
from qualia_core.typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, dict_items

    from numpy._typing._array_like import NDArray

logger: logging.Logger = logging.getLogger(__name__)

class EuroSAT(RawDataset):
    """EuroSAT Land Use and Land Cover Classification with Sentinel-2,.

    Challenge of land use and land cover classification using Sentinel-2 satellite images.
    The Sentinel-2 satellite images are openly and freely accessible provided in the Earth observation program Copernicus.
    Sentinel-2 satellite images covering 13 spectral bands.
    10 classes with in total 27,000 labeled and geo-referenced images.
    The paper proposed network that achieved an overall classification accuracy of 98.57%.
    The geo-referenced dataset EuroSAT is made publicly available here [https://github.com/phelber/EuroSAT].
    ---
    10 Classes - 27000 images
    1. Annual Crop              - 3000 images
    2. Forest                   - 3000 images
    3. Herbaceous Vegetation    - 3000 images
    4. Highway                  - 2500 images
    5. Industrial Buildings     - 2500 images
    6. Pasture                  - 2000 images
    7. Permanent Crop           - 2500 images
    8. Residential Buildings    - 3000 images
    9. River                    - 2500 images
    10. Sea and Lake            - 3000 images
    """

    def __init__(self, path: str = '', variant: str = 'MS', dtype: str = 'float32') -> None:
        """Variant available Multi Spectral (MS) or RGB, Only MS inmplemented so far."""
        if variant not in ['MS', 'RGB']:
            msg: str = f"Invalid variant '{variant}'. Choose 'MS' or 'RGB'."
            raise ValueError(msg)
        super().__init__()  # Set up the basic RawDataset structure
        self.__path = Path(path)  # Convert string path to a proper Path object
        self.__dtype: str = dtype
        self.__variant: str = variant    # Store which variant we want to use
        self.sets.remove('valid')   # Tell Qualia we won't use a validation set

    def _dataset_info(self) -> dict[str, int]:
        """Provide information about the dataset.

        This is like giving a brief overview of what our dataset contains:
        - How many classes (types of things) are there?
        - What are the names of these classes?
        - How many images are in each class?

        This helps us understand what we have before we start using it.
        """
        start: float = time.time()

        images_path: Path = self.__path / self.__variant
        # get the number of folders, which is the number of classes and the name the names of the classes
        class_names: list[str] = sorted([d.name for d in images_path.iterdir() if d.is_dir()])
        self.class_idx: dict[str, int] = {name: idx for idx, name in enumerate(class_names)}
        len(class_names)

        # for each class, get the number of elements
        class_counts: dict[str, int] = dict.fromkeys(class_names, 0)
        for class_name in class_names:
            class_path: Path = images_path / class_name
            if not class_path.is_dir():
                logger.warning('Skipping %s, not a directory', class_path)
                continue
            if self.__variant == 'MS':
                count: int = len(list(class_path.glob('*.tif')))
            elif self.__variant == 'RGB':
                count = len(list(class_path.glob('*.jpg')))
            else:
                raise ValueError(f"Unsupported variant '{self.__variant}'. Use 'MS' or 'RGB'.")
            class_counts[class_name] = count
        logger.info('_dataset_info() Elapsed: %s s', time.time() - start)

        return class_counts

    def _generate_test_train_split(self) -> None:
        start: float = time.time()
        class_counts: dict[str, int] = self._dataset_info()  # Get info about the dataset
        if exists(self.__path/'test_idx.npy') and exists(self.__path/'train_idx.npy'):
            logger.info('Test/train split already exists, loading from files.')
            self.test_idx: dict[str, np.ndarray] = np.load(self.__path/'test_idx.npy', allow_pickle=True).item()
            self.train_idx: dict[str, np.ndarray] = np.load(self.__path/'train_idx.npy', allow_pickle=True).item()
            logger.info('_generate_test_train_split() Elapsed: %s s', time.time() - start)
            return

        train_test_ratio = 0.8
        test_idx: dict[str, NDArray[Any,Any]] = {name: np.array([], dtype=int) for name in class_counts}
        train_idx: dict[str, NDArray[Any]] = {name: np.array([], dtype=int) for name in class_counts}

        for class_name, count in class_counts.items():
            test_idx[class_name] = random.shared.generator.choice(
                np.arange(count) + 1,
                size=int(count * (1 - train_test_ratio)),
                replace=False,
            )
            train_idx[class_name] = np.setdiff1d(
                np.arange(count)+1,
                test_idx[class_name],
            )
        logger.info('Generated test/train split: %s', class_counts)

        # Save the indices for later use
        with Path.open(self.__path / 'test_idx.npy', 'wb') as f:
            np.save(f, test_idx)
        with Path.open(self.__path / 'train_idx.npy', 'wb') as f:
            np.save(f, train_idx)
        self.test_idx   = test_idx
        self.train_idx  = train_idx
        logger.info('_generate_test_train_split() Elapsed: %s s', time.time() - start)
        return

    def __load_data(self, *, train:bool=True) -> RawData:
        """Load and preprocess data files.

        This is where we:
        1. Read our raw data files
        2. Format them how Qualia expects
        3. Make sure values are in the right range

        It's like taking ingredients and preparing them for cooking:
        - Reading the files is like getting ingredients from containers
        - Reshaping is like cutting them to the right size
        - Normalizing is like measuring out the right amounts
        """
        start: float = time.time()
        self._generate_test_train_split()

        train_x_list: list[npt.NDArray[np.uint16]] = []
        train_y_list: list[int] = []
        images_path: Path = self.__path / self.__variant
        items: dict_items[str, NDArray] = self.train_idx.items() if train else self.test_idx.items()

        for class_name, indices in items:
            class_path: Path = images_path / class_name
            if not class_path.is_dir():
                logger.warning('Skipping %s, not a directory', class_path)
                continue
            for idx in indices:
                if self.__variant == 'MS':
                    filepath: Path = class_path / f'{class_name}_{idx:d}.tif'
                elif self.__variant == 'RGB':
                    filepath: Path = class_path / f'{class_name}_{idx:d}.jpg'
                else:
                    raise ValueError(f"Unsupported variant '{self.__variant}'. Use 'MS' or 'RGB'.")
                if not filepath.is_file():
                    logger.warning('Skipping %s, not a file', filepath)
                    continue

                if self.__variant == 'MS':
                    data: NDArray[logging.Any, np.dtype[logging.Any]] = tiff.imread(filepath) # data is shape 64, 64, 13
                elif self.__variant == 'RGB':
                    data: NDArray[logging.Any, np.dtype[logging.Any]] = np.array(Image.open(filepath))
                else:
                    raise ValueError(f"Unsupported variant '{self.__variant}'. Use 'MS' or 'RGB'.")
                train_x_list.append(data)
                train_y_list.append(self.class_idx[class_name])  # Use the class index for labels
        # Convert lists to numpy arrays
        train_x_uint16 = np.array(train_x_list, dtype=np.uint16)

        if self.__variant == 'MS':
            train_x_uint16 = train_x_uint16.reshape((train_x_uint16.shape[0], 64, 64, 13))  # N, C, H, W
        elif self.__variant == 'RGB':
            train_x_uint16 = train_x_uint16.reshape((train_x_uint16.shape[0], 64, 64, 3))
        else:
            raise ValueError(f"Unsupported variant '{self.__variant}'. Use 'MS' or 'RGB'.")

        train_x: NDArray[Any] = train_x_uint16.astype(self.__dtype) # N, H, W, C
        train_y: NDArray[Any] = np.array(train_y_list, dtype=np.int64)  # Convert labels to numpy array
        logger.info('__load_train() Elapsed: %s s', time.time() - start)
        return RawData(train_x, train_y)

    def __call__(self) -> RawDataModel:
        """Load and prepare the complete dataset.

        This is our main kitchen where we:
        1. Load all our data
        2. Organize it into training and test sets
        3. Package it in Qualia's preferred containers
        4. Add helpful information for debugging
        """
        logger.info('Loading EuroSAT dataset from %s', self.__path)

        # Package everything in Qualia's containers
        return RawDataModel(
            sets=RawDataSets(
                train=self.__load_data(train=True),
                test=self.__load_data(train=False),
            ),
            name=self.name,
        )
