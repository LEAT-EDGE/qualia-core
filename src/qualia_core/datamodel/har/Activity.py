from dataclasses import dataclass, field

import numpy as np
import numpy.typing

from qualia_core.datamodel.TimeSample import TimeSample

from .Activities import Activities


@dataclass
class Activity:
    kind: Activities
    samples: list[TimeSample] = field(default_factory=list)

    def get_raw_array(self) -> list[list[numpy.typing.NDArray[np.float32]]]:
        return [s.get_raw_array() for s in self.samples]
