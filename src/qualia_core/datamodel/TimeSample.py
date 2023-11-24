from dataclasses import dataclass, field

import numpy as np
import numpy.typing

from qualia_core.datamodel.sensor.Sensor import Sensor


@dataclass
class TimeSample:
    t: float
    sensors: list[Sensor] = field(default_factory=list)

    def get_raw_array(self) -> list[numpy.typing.NDArray[np.float32]]:
        return [v for s in self.sensors for v in s.get_raw_array()]
