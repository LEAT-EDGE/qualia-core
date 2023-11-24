from dataclasses import asdict, astuple, dataclass, fields
from typing import Any

import numpy as np
import numpy.typing


@dataclass
class Sensor:
    @classmethod
    def fieldnames(cls) -> tuple[str, ...]:
        return tuple(f.name for f in fields(cls))

    def asdict(self) -> dict[str, Any]:
        """Warning: dataclasses.asdict() and dataclasses.astuple() are recursive and may copy."""
        return asdict(self)

    def get_raw_array(self) -> numpy.typing.NDArray[np.float32]:
        return np.array(astuple(self), dtype=np.float32)
