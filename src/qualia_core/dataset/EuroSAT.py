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

import json
import logging  # logging: For keeping track of what our dataset is doing
import sys
import time
from pathlib import Path  # Path: Makes file handling consistent across operating systems
from typing import Any

import numpy as np  # numpy: For efficient array operations on our data

from qualia_core import random  # Generator: For generating random numbers, useful for splitting data into training and test sets
from qualia_core.datamodel.RawDataModel import RawData, RawDataModel, RawDataSets
from qualia_core.dataset.RawDataset import RawDataset

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

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

    def __init__(self,
                 path: str = '',
                 variant: str = 'MS',
                 dtype: str = 'float32',
                 train_test_ratio: float = 0.8) -> None:
        """Instantiate the EuroSAT dataset loader.

        :param path: Dataset source path
        :param variant: ``'MS'`` (Multi Spectral) or ``'RGB'``, Only MS inmplemented so far.
        :param dtype: Data type for the input vectors
        """
        super().__init__()  # Set up the basic RawDataset structure
        self.__path = Path(path)  # Convert string path to a proper Path object

        if variant == 'MS':
            self.__suffix = 'tif'
            self.__channels = 13
        elif variant == 'RGB':
            self.__suffix = 'jpg'
            self.__channels = 3
        else:
            logger.error("Unsupported variant '%s'. Use 'MS' or 'RGB'.", variant)
            raise ValueError

        self.__variant = variant    # Store which variant we want to use
        self.__dtype = dtype
        self.__train_test_ratio = train_test_ratio
        self.sets.remove('valid')   # Tell Qualia we won't use a validation set

    def _dataset_info(self) -> tuple[dict[str, int], dict[str, int]]:
        """Provide information about the dataset.

        This is like giving a brief overview of what our dataset contains:
        - How many classes (types of things) are there?
        - What are the names of these classes?
        - How many images are in each class?

        This helps us understand what we have before we start using it.
        """
        start = time.time()

        images_path = self.__path / self.__variant
        # get the number of folders, which is the number of classes and the name the names of the classes
        class_names: list[str] = sorted([d.name for d in images_path.iterdir() if d.is_dir()])
        class_idx = {name: idx for idx, name in enumerate(class_names)}

        # for each class, get the number of elements
        class_counts = dict.fromkeys(class_names, 0)
        for class_name in class_names:
            class_path = images_path / class_name
            if not class_path.is_dir():
                logger.warning('Skipping %s, not a directory', class_path)
                continue
            class_counts[class_name] = len(list(class_path.glob(f'*.{self.__suffix}')))
        logger.info('_dataset_info() Elapsed: %s s', time.time() - start)

        return class_counts, class_idx

    def _generate_test_train_split(self, class_counts: dict[str, int]) -> tuple[dict[str, np.ndarray[Any, np.dtype[np.int64]]],
                                                                                dict[str, np.ndarray[Any, np.dtype[np.int64]]]]:
        start = time.time()

        train_idx = {name: np.array([], dtype=np.int64) for name in class_counts}
        test_idx = {name: np.array([], dtype=np.int64) for name in class_counts}

        for class_name, count in class_counts.items():
            test_idx[class_name] = random.shared.generator.choice(
                np.arange(count) + 1,
                size=int(count * (1 - self.__train_test_ratio)),
                replace=False,
            ).tolist()
            train_idx[class_name] = np.setdiff1d(
                np.arange(count)+1,
                test_idx[class_name],
            ).tolist()
        logger.info('Generated test/train split: %s', class_counts)

        # Save the indices for later use
        with Path.open(self.__path / 'test_idx.json', 'w') as f:
            json.dump(test_idx, f, indent='  ')
        with Path.open(self.__path / 'train_idx.json', 'w') as f:
            json.dump(train_idx, f, indent='  ')

        logger.info('_generate_test_train_split() Elapsed: %s s', time.time() - start)
        return train_idx, test_idx

    def __load_data(self, *, class_idx: dict[str, int], set_idx: dict[str, np.ndarray[Any, np.dtype[np.int64]]]) -> RawData:
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
        import imageio

        start = time.time()

        train_x_list: list[np.ndarray[Any, np.dtype[np.uint16]]] = []
        train_y_list: list[int] = []

        for class_name, indices in set_idx.items():
            class_path = self.__path / self.__variant / class_name
            if not class_path.is_dir():
                logger.warning('Skipping %s, not a directory', class_path)
                continue
            for idx in indices:
                filepath = class_path / f'{class_name}_{idx:d}.{self.__suffix}'
                if not filepath.is_file():
                    logger.warning('Skipping %s, not a file', filepath)
                    continue

                data = imageio.v3.imread(filepath)

                train_x_list.append(data)
                train_y_list.append(class_idx[class_name])  # Use the class index for labels

        # Convert lists to numpy arrays
        train_x_uint16 = np.array(train_x_list, dtype=np.uint16)

        train_x_uint16 = train_x_uint16.reshape((train_x_uint16.shape[0], 64, 64, self.__channels))

        train_x = train_x_uint16.astype(self.__dtype) # N, H, W, C
        train_y = np.array(train_y_list, dtype=np.int64)  # Convert labels to numpy array
        logger.info('__load_train() Elapsed: %s s', time.time() - start)
        return RawData(train_x, train_y)

    @override
    def __call__(self) -> RawDataModel:
        """Load and prepare the complete dataset.

        This is our main kitchen where we:
        1. Load all our data
        2. Organize it into training and test sets
        3. Package it in Qualia's preferred containers
        4. Add helpful information for debugging
        """
        logger.info('Loading EuroSAT dataset from %s', self.__path)

        class_counts, class_idx = self._dataset_info()

        if (self.__path/'test_idx.json').exists() and (self.__path/'train_idx.json').exists():
            logger.info('Test/train split already exists, loading from files.')
            with (self.__path/'train_idx.json').open() as f:
                train_idx = json.load(f)
            with (self.__path/'test_idx.json').open() as f:
                test_idx = json.load(f)
        else:
            train_idx, test_idx = self._generate_test_train_split(class_counts=class_counts)

        # Package everything in Qualia's containers
        return RawDataModel(
            sets=RawDataSets(
                train=self.__load_data(class_idx=class_idx, set_idx=train_idx),
                test=self.__load_data(class_idx=class_idx, set_idx=test_idx),
            ),
            name=self.name,
        )
