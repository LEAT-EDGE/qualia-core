from __future__ import annotations

from qualia_core.evaluation.STM32CubeAI import STM32CubeAI as STM32CubeAIBase


class STM32CubeAI(STM32CubeAIBase):
    def __init__(self,
        dev: str = '/dev/ttyACM0',
        baudrate: int = 921600) -> None:
        super().__init__(mode='stm32', stm32cubeai_args=('--desc', f'{dev}:{baudrate}'))
