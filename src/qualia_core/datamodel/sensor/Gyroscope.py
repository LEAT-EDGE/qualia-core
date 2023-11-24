from dataclasses import dataclass

from .Sensor import Sensor

@dataclass
class Gyroscope(Sensor):
    x: float
    y: float
    z: float
