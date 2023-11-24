from dataclasses import dataclass

from .Sensor import Sensor

@dataclass
class GPS(Sensor):
    latitude: float
    longitude: float
    altitude: float
    speed: float
    bearing: float
    accuracy: float
