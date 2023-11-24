from dataclasses import dataclass

from .Sensor import Sensor

@dataclass
class Accelerometer(Sensor):
    x: float
    y: float
    z: float
