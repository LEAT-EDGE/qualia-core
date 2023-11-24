from dataclasses import dataclass

from qualia_core.datamodel.har.Activity import Activity


@dataclass
class Subject:
    name: str
    activities: list[Activity]
    part: str = ''
