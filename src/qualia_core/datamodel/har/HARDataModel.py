from qualia_core.datamodel import DataModel

from . import Subject


class HARDataModel(DataModel[list[Subject]]):
    sets: DataModel.Sets[list[Subject]]

    def __init__(self, sets: DataModel.Sets[list[Subject]], name: str = 'HAR') -> None:
        super().__init__(sets, name)
        self.name = name
