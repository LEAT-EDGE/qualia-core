from pathlib import Path
from typing import Callable

import dill
from filelock import FileLock
import pytest


def uci_har_preprocess_data() -> dict:
    import qualia_core.utils.config
    from qualia_core import main
    configf, configname = qualia_core.utils.config.parse_config(Path('conf')/'tests'/'UCI-HAR_ResNetv1_train_float32.toml')
    config = qualia_core.utils.config.validate_config_dict(configf)
    return main.qualia('preprocess_data', config, configname)

def uci_har_resnet_train_float32(fixture_uci_har_preprocess_data: Callable[[], dict]) -> dict:
    from qualia_core import main
    import qualia_core.utils.config
    configf, configname = qualia_core.utils.config.parse_config(Path('conf')/'tests'/'UCI-HAR_ResNetv1_train_float32.toml')
    config = qualia_core.utils.config.validate_config_dict(configf)
    return {**main.qualia('train', config, configname), **fixture_uci_har_preprocess_data}

def uci_har_resnet_train_int16(fixture_uci_har_preprocess_data: Callable[[], dict],
                               fixture_uci_har_resnet_train_float32: Callable[[], dict]) -> dict:
    from qualia_core import main
    import qualia_core.utils.config
    configf, configname = qualia_core.utils.config.parse_config(Path('conf')/'tests'/'UCI-HAR_ResNetv1_train_int16.toml')
    config = qualia_core.utils.config.validate_config_dict(configf)
    return {**main.qualia('train', config, configname), **fixture_uci_har_preprocess_data}

def uci_har_resnet_train_int8(fixture_uci_har_preprocess_data: Callable[[], dict],
                              fixture_uci_har_resnet_train_float32: Callable[[], dict]) -> dict:
    from qualia_core import main
    import qualia_core.utils.config
    configf, configname = qualia_core.utils.config.parse_config(Path('conf')/'tests'/'UCI-HAR_ResNetv1_train_int8.toml')
    config = qualia_core.utils.config.validate_config_dict(configf)
    return {**main.qualia('train', config, configname), **fixture_uci_har_preprocess_data}

def xdist_shared(worker_id: str, tmp_path_factory, name: str, callback: Callable[[], dict]):
    if worker_id == 'master':
        return callback()

    root_tmp_dir = tmp_path_factory.getbasetemp().parent

    fn = root_tmp_dir / f'{name}.json'
    with FileLock(str(fn) + '.lock'):
        if fn.is_file():
            from qualia_core.utils.TensorFlowInitializer import TensorFlowInitializer
            # workaround to prevent deserialization from loading a keras/TF module without calling TensorFlowInitializer
            TensorFlowInitializer()(seed=2, reserve_gpu=False, gpu_memory_growth=True)

            with fn.open('rb') as f:
                data = dill.load(f)
        else:
            data = callback()
            with fn.open('wb') as f:
                dill.dump(data, f, protocol=dill.HIGHEST_PROTOCOL)
    return data

@pytest.fixture(scope='package')
def fixture_uci_har_preprocess_data(worker_id: str, tmp_path_factory) -> dict:
    return xdist_shared(worker_id, tmp_path_factory, 'fixture_uci_har_preprocess_data', uci_har_preprocess_data)

@pytest.fixture(scope='package')
def fixture_uci_har_resnet_train_float32(worker_id: str,
                                         tmp_path_factory,
                                         fixture_uci_har_preprocess_data: Callable[[], dict]) -> dict:
    return xdist_shared(worker_id, tmp_path_factory, 'fixture_uci_har_resnet_train_float32', lambda: uci_har_resnet_train_float32(fixture_uci_har_preprocess_data))


@pytest.fixture(scope='package')
def fixture_uci_har_resnet_train_int16(worker_id: str,
                                       tmp_path_factory,
                                       fixture_uci_har_preprocess_data: Callable[[], dict],
                                       fixture_uci_har_resnet_train_float32: Callable[[], dict]) -> dict:
    return xdist_shared(worker_id, tmp_path_factory, 'fixture_uci_har_resnet_train_int16', lambda:
                        uci_har_resnet_train_int16(fixture_uci_har_preprocess_data, fixture_uci_har_resnet_train_float32))

@pytest.fixture(scope='package')
def fixture_uci_har_resnet_train_int8(worker_id: str,
                                      tmp_path_factory,
                                      fixture_uci_har_preprocess_data: Callable[[], dict],
                                      fixture_uci_har_resnet_train_float32: Callable[[], dict]) -> dict:
    return xdist_shared(worker_id, tmp_path_factory, 'fixture_uci_har_resnet_train_int8', lambda:
                        uci_har_resnet_train_int8(fixture_uci_har_preprocess_data, fixture_uci_har_resnet_train_float32))
