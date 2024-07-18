from dataclasses import asdict, astuple, dataclass, field, fields
from typing import Any, NamedTuple


class StatsFields(NamedTuple):
    name: str
    i: int
    quantization: str
    params: int
    mem_params: int
    accuracy: float # between 0 and 1
    avg_time: float # in seconds
    rom_size: int # in bytes
    ram_size: int # in bytes
    metrics: dict[str, float]

@dataclass
class Stats:
    name: str = ''
    i: int = -1
    quantization: str = 'float32'
    params: int = -1
    mem_params: int = -1
    accuracy: float = -1 # between 0 and 1
    avg_time: float = -1 # in seconds
    rom_size: int = -1 # in bytes
    ram_size: int = -1 # in bytes
    metrics: dict[str, float] = field(default_factory=dict)

    @classmethod
    def fieldnames(cls) -> tuple[str, ...]:
        return tuple(f.name for f in fields(Stats))

    def asdict(self) -> dict[str, Any]:
        return asdict(self)

    def astuple(self) -> tuple[Any, ...]:
        return astuple(self)

    def asnamedtuple(self) -> StatsFields:
        return StatsFields(**asdict(self))
