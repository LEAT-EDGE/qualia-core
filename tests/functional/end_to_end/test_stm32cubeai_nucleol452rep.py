from pathlib import Path
from typing import Callable

import pytest


class TestSTM32CubeAINucleoL452REP:
    '''Must be executed from command line with PYTHONHASHSEED environment variable defined to the "seed" value of the "bench"
    section of the configuration file.
    '''

    @pytest.mark.dependency()
    @pytest.mark.xdist_group(name='uci_har_resnet_stm32cubeai_nucleol452rep')
    def test_uci_har_resnet_stm32cubeai_nucleol452rep_float32_prepare_deploy(self, fixture_uci_har_resnet_train_float32: Callable[[], dict]) -> None:
        from qualia_core import main
        import qualia_core.utils.config
        configf, configname = qualia_core.utils.config.parse_config(Path('conf')/'tests'/'UCI-HAR_ResNetv1_STM32CubeAI_NucleoL452REP_float32.toml')
        config = qualia_core.utils.config.validate_config_dict(configf)
        loggers = fixture_uci_har_resnet_train_float32

        assert 'learningmodel' in loggers

        assert len(loggers['learningmodel']) == 1

        assert loggers['learningmodel'][0].name == 'uci-har_resnetv1_8'
        assert loggers['learningmodel'][0].i == 1
        assert loggers['learningmodel'][0].params == 1150
        assert loggers['learningmodel'][0].mem_params == 4600
        assert loggers['learningmodel'][0].accuracy >= 0.85

        loggers = main.qualia('prepare_deploy', config, configname)

        assert 'prepare_deploy' in loggers

        assert len(loggers['prepare_deploy']) == 1
        assert loggers['prepare_deploy'][0].name == 'uci-har_resnetv1_8'
        assert loggers['prepare_deploy'][0].quantize == 'float32'
        assert loggers['prepare_deploy'][0].optimize == 'cmsis-nn'
        assert loggers['prepare_deploy'][0].compress == 1

    @pytest.mark.dependency(depends=['TestSTM32CubeAINucleoL452REP::test_uci_har_resnet_stm32cubeai_nucleol452rep_float32_prepare_deploy'])
    @pytest.mark.xdist_group(name='uci_har_resnet_stm32cubeai_nucleol452rep')
    @pytest.mark.deploy()
    def test_uci_har_resnet_stm32cubeai_nucleol452rep_float32_deploy_and_evaluate(self):
        from qualia_core import main
        import qualia_core.utils.config
        configf, configname = qualia_core.utils.config.parse_config(Path('conf')/'tests'/'UCI-HAR_ResNetv1_STM32CubeAI_NucleoL452REP_float32.toml')
        config = qualia_core.utils.config.validate_config_dict(configf)
        loggers = main.qualia('deploy_and_evaluate', config, configname)

        assert len(loggers['evaluate']) == 1

        assert loggers['evaluate'][0].name == 'uci-har_resnetv1_8'
        assert loggers['evaluate'][0].i == 1
        assert loggers['evaluate'][0].quantization == 'float32'
        assert loggers['evaluate'][0].params == 1150
        assert loggers['evaluate'][0].mem_params == 4600
        assert loggers['evaluate'][0].accuracy >= 0.85
        assert loggers['evaluate'][0].avg_time <= 0.04
        assert loggers['evaluate'][0].rom_size <= 60000

    @pytest.mark.dependency()
    @pytest.mark.xdist_group(name='uci_har_resnet_stm32cubeai_nucleol452rep')
    def test_uci_har_resnet_stm32cubeai_nucleol452rep_int8_prepare_deploy(self, fixture_uci_har_resnet_train_int8: Callable[[], dict]) -> None:
        from qualia_core import main
        import qualia_core.utils.config
        configf, configname = qualia_core.utils.config.parse_config(Path('conf')/'tests'/'UCI-HAR_ResNetv1_STM32CubeAI_NucleoL452REP_int8.toml')
        config = qualia_core.utils.config.validate_config_dict(configf)
        loggers = fixture_uci_har_resnet_train_int8

        assert 'learningmodel' in loggers

        assert len(loggers['learningmodel']) == 2
        assert loggers['learningmodel'][0].name == 'uci-har_resnetv1_8'
        assert loggers['learningmodel'][1].name == 'uci-har_resnetv1_8_q8_force_qoff_e1_LSQFalse'
        assert loggers['learningmodel'][0].i == 1
        assert loggers['learningmodel'][1].i == 1
        assert loggers['learningmodel'][0].params == 1150
        assert loggers['learningmodel'][1].params == 1150
        assert loggers['learningmodel'][0].mem_params == 4600 # Model is float32 after training
        assert loggers['learningmodel'][1].mem_params == 1150 # Model is(virtually) int8 after PTQ
        assert loggers['learningmodel'][0].accuracy >= 0.85
        assert loggers['learningmodel'][1].accuracy >= 0.85


        loggers = main.qualia('prepare_deploy', config, configname)

        assert 'prepare_deploy' in loggers

        assert len(loggers['prepare_deploy']) == 1
        assert loggers['prepare_deploy'][0].name == 'uci-har_resnetv1_8_q8_force_qoff_e1_LSQFalse'
        assert loggers['prepare_deploy'][0].quantize == 'int8'
        assert loggers['prepare_deploy'][0].optimize == 'cmsis-nn'
        assert loggers['prepare_deploy'][0].compress == 1

    @pytest.mark.dependency(depends=['TestSTM32CubeAINucleoL452REP::test_uci_har_resnet_stm32cubeai_nucleol452rep_int8_prepare_deploy'])
    @pytest.mark.xdist_group(name='uci_har_resnet_stm32cubeai_nucleol452rep')
    @pytest.mark.deploy()
    def test_uci_har_resnet_stm32cubeai_nucleol452rep_int8_deploy_and_evaluate(self):
        from qualia_core import main
        import qualia_core.utils.config
        configf, configname = qualia_core.utils.config.parse_config(Path('conf')/'tests'/'UCI-HAR_ResNetv1_STM32CubeAI_NucleoL452REP_int8.toml')
        config = qualia_core.utils.config.validate_config_dict(configf)
        loggers = main.qualia('deploy_and_evaluate', config, configname)

        assert len(loggers['evaluate']) == 1

        assert loggers['evaluate'][0].name == 'uci-har_resnetv1_8_q8_force_qoff_e1_LSQFalse'
        assert loggers['evaluate'][0].i == 1
        assert loggers['evaluate'][0].quantization == 'int8'
        assert loggers['evaluate'][0].params == 1150
        assert loggers['evaluate'][0].mem_params == 1150
        assert loggers['evaluate'][0].accuracy >= 0.85
        assert loggers['evaluate'][0].avg_time <= 0.02
        assert loggers['evaluate'][0].rom_size <= 80000
