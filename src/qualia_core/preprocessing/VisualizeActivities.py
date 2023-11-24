from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .Preprocessing import Preprocessing

class VisualizeActivities(Preprocessing):
    def __init__(self, activities, noconvert: bool=False):
        from qualia_core.datamodel.har import Activities
        for i, activity in enumerate(activities):
            if not noconvert and isinstance(activity, str):
                activities[i] = Activities[activity]
        self.__activities = activities


    def __call__(self, datamodel):
        outpath = Path('out')/'visualization'/'activities'/datamodel.name

        fig = plt.figure()
        for sname, s in datamodel.sets:
            for subject in s:
                outdir = outpath/sname/subject.name
                outdir.mkdir(parents=True, exist_ok=True)

                to_plot = {}
                
                for i, activity in enumerate(subject.activities):
                    if activity.kind in self.__activities:
                        to_plot.setdefault(activity.kind, []).append([i, activity])

                for kind, acts in to_plot.items():

                    outfile = outdir/f'{kind.name}.pdf'

                    pp = PdfPages(outfile)

                    for (activity_i, act) in acts:
                        samples = np.array(act.get_raw_array())
                        samples = samples.reshape((samples.shape[0], -1))
                        for i in range(samples.shape[-1]):
                            p = plt.plot(samples[:,i])

                        plt.suptitle(activity_i)

                        pp.savefig(fig)
                        fig.clf()
                    pp.close()
        return datamodel
