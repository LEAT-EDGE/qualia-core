from dataclasses import dataclass

from .Sensor import Sensor

@dataclass
class Barometer(Sensor):
    p: float
