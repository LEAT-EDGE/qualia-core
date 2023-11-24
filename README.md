# Qualia Core (formerly MicroAI)
End-to-end training, quantization and deployment framework for deep neural networks on microcontrollers.

Repository should be cloned with `--recursive` to get TFLite Micro and its dependencies.

## Dependencies

Python:
```
numpy
scikit-learn
tomlkit
colorful
gitpython
```

### Dataset

#### GTSRB

Python:
```
imageio
scikit-image
```

### Training

#### TensorFlow

Python:
```
tensorflow
tensorflow_addons
```

#### PyTorch

Python:
```
pytorch
pytorch_lightning
```

### Deployment

#### Embedded targets

##### SparkFun Edge 

Python:
```
pycryptodome
```

##### Nucleo-L452RE-P
System:
```
stm32cubeide
stm32cubeprog
```

#### Embedded frameworks

##### STM32Cube.AI

STM32CubeIDE extension pack:
```
X-CUBE-AI == 5.2.0
```

##### TensorFlow Lite Micro

System:
```
arm-none-eabi-binutils
arm-none-eabi-gcc
arm-none-eabi-newlib
libopenexr-dev
wget
```

##### Qualia-CodeGen
Python:
```
jinja2
```

System:
```
arm-none-eabi-binutils
arm-none-eabi-gcc
arm-none-eabi-newlib
```

### Evaluation
Python:
```
pyserial
```

## Usage

If Qualia installed with pip, you can run the `qualia` command directly. Otherwise run `PYTHONPATH=. ./bin/qualia <config.toml> <action>` from the qualia directory.

### Dataset pre-processing
```
qualia <config.toml> preprocess_data
```

### Training
```
qualia <config.toml> train
```

### Prepare deployment (generate firmware)
```
qualia <config.toml> prepare_deploy
```

### Deploy and evaluate
```
qualia <config.toml> deploy_and_evaluate
```

### Run test suite
```
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=2 python -m unittest discover qualia/tests
```

## Included support for datasets, learning framework, neural networks, embedded frameworks and targets

### Datasets
- [UCI HAR](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)
- [GTSRB](https://benchmark.ini.rub.de/)
- [Written and Spoken MNIST](https://zenodo.org/record/3515935)

### Learning frameworks
- TensorFlow.Keras
- PyTorch

### Neural networks
- MLP
- CNN (1D&2D)
- Resnetv1 (1D&2D)

### Embedded frameworks
- STM32Cube.AI
- TensorFlow Lite for Microcontrollers
- Qualia-CodeGen

### Targets
- Nucleo-L452RE-P
- SparkFun Edge

## Reference & Citation

[Quantization and Deployment of Deep Neural Networks on Microcontrollers](https://www.mdpi.com/1424-8220/21/9/2984), Pierre-Emmanuel Novac, Ghouthi Boukli Hacene, Alain Pegatoquet, Benoît Miramond and Vincent Gripon, Sensors, 2021.

```
@article{qualia,
	author = {Novac, Pierre-Emmanuel and Boukli Hacene, Ghouthi and Pegatoquet, Alain and Miramond, Benoît and Gripon, Vincent},
	title = {Quantization and Deployment of Deep Neural Networks on Microcontrollers},
	journal = {Sensors},
	volume = {21},
	year = {2021},
	number = {9},
	article-number = {2984},
	url = {https://www.mdpi.com/1424-8220/21/9/2984},
	issn = {1424-8220},
	doi = {10.3390/s21092984}
}
```
