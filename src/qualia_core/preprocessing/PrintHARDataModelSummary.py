from .Preprocessing import Preprocessing

from qualia_core.utils.logger import TextLogger

class PrintHARDataModelSummary(Preprocessing):
    log = TextLogger(name=__name__)

    def __print_subjects_summary(self, name, s, activitylabels):
        msg = f'Set: {name}\n'
        msg += '      '
        for act in activitylabels:
            msg += f'  {act.name}'
        msg += '  TOTAL\n'

        ss = sorted(s, key=lambda s: int(s.name[1:])) # Sort by subject number (without T prefix)

        total_a_count = [0] * len(activitylabels)

        for subject in ss:
            msg += ' ' * (5 - len(subject.name))
            msg += f'{subject.name}: '
            a_count = [0] * len(activitylabels)
            for a in subject.activities:
                a_count[activitylabels.index(a.kind)] += len(a.get_raw_array())
                total_a_count[activitylabels.index(a.kind)] += len(a.get_raw_array())

            for a, c in enumerate(a_count):
                msg += ' ' * (len(activitylabels[a].name) - len(str(c)) + 1)
                msg += f'{c} '
            total = sum(a_count)
            msg += ' ' * (len('TOTAL') - len(str(total)) + 1)
            msg += f'{total}\n'

        # Total of activities across all subjects
        msg += 'Total: '

        for a, c in enumerate(total_a_count):
            msg += ' ' * (len(activitylabels[a].name) - len(str(c)) + 1)
            msg += f'{c} '
        total = sum(total_a_count)
        msg += ' ' * (len('TOTAL') - len(str(total)) + 1)
        msg += f'{total}\n'

        print(msg, end='')
        self.log(msg)

        return total_a_count

    def __print_set_summary(self, sets_total, activitylabels):
        msg = '      '
        for act in activitylabels:
            msg += f'  {act.name}'
        msg += '  TOTAL\n'

        global_a_total = [sum([s[i] for s in sets_total.values()]) for i, _ in enumerate(activitylabels)]
        global_total = sum(global_a_total)
        sets_percentage = {name: [s[i] / global_a_total[i] for i, _ in enumerate(activitylabels)] for name, s in sets_total.items()}

        for (name, s), stot in zip(sets_percentage.items(), sets_total.values()):
            msg += ' ' * (5 - len(name))
            msg += f'{name}: '

            for a, c in enumerate(s):
                c = int(c * 100)
                msg += ' ' * (len(activitylabels[a].name) - len(str(c)))
                msg += f'{c}% '

            total = int(sum(stot) * 100 / global_total)
            msg += ' ' * (len('TOTAL') - len(str(total)))
            msg += f'{total}%\n'

        print(msg, end='')
        self.log(msg)

    def __call__(self, datamodel):
        # First generate the class numbers according to the activities present in the dataset
        # Duplicated from DatamodelConverter since they are not generated yet
        activitylabels = set()
        for sname, s in datamodel.sets:
            for subject in s:
                for activity in subject.activities:
                    activitylabels.add(activity.kind)
        activitylabels = sorted(activitylabels)

        sets_total = {}
        for name, s in datamodel:
            sets_total[name] = self.__print_subjects_summary(name, s, activitylabels)
        self.__print_set_summary(sets_total, activitylabels)

        return datamodel
