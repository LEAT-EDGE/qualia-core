from enum import auto

from qualia_core.utils import StrEnum

from enum import IntEnum

class Activities(IntEnum): #(StrEnum):
    NULL = auto()
    STANDING = auto()
    STAND_TO_SIT = auto()
    SITTING = auto()
    SIT_TO_STAND = auto()
    WALKING = auto()
    STOPPING = auto()
    LYING = auto()
    STAND_TO_LIE = auto()
    LIE_TO_STAND = auto()
    STAND_TO_SQUAT = auto()
    SQUAT_TO_STAND = auto()
    SQUATTING = auto()
    WALKING_DOWNSTAIRS = auto()
    WALKING_UPSTAIRS = auto()

    # Used in RealLife_HAR
    INACTIVE = auto()
    ACTIVE = auto()
    DRIVING = auto()

    # Used in EllcieHAR
    RUNNING = auto()
    DRINKING = auto()
    SIT_TO_LIE = auto()
    LIE_TO_SIT = auto()
