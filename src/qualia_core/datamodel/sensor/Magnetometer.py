from dataclasses import dataclass

from .Sensor import Sensor

@dataclass
class Magnetometer(Sensor):
    x: float
    y: float
    z: float
