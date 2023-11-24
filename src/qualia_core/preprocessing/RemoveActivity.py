from .Preprocessing import Preprocessing

class RemoveActivity(Preprocessing):
    def __init__(self, activities, noconvert: bool=False):
        from qualia_core.datamodel.har import Activities
        for i, activity in enumerate(activities):
            if not noconvert and isinstance(activity, str):
                activities[i] = Activities[activity]
        self.__activities = activities

    def __call__(self, datamodel):
        for _, subjectset in datamodel:
            for subject in subjectset:
                subject.activities = [activity for activity in subject.activities if activity.kind not in self.__activities]
        return datamodel
