from enum import Enum

# From https://github.com/irgeek/StrEnum
class StrEnum(str, Enum):
    def __str__(self):
        return self.value

    def _generate_next_value_(name, *_):
        return name
