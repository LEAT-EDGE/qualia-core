from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .Preprocessing import Preprocessing

class VisualizeWindows(Preprocessing):
    def __init__(self, labels: list):
        self.__labels = labels

    def __call__(self, datamodel):
        outpath = Path('out')/'visualization'/'windows'/datamodel.name

        fig = plt.figure()
        for sname, s in datamodel:
            outdir = outpath/sname
            outdir.mkdir(parents=True, exist_ok=True)

            pdffiles = {}
            for label in self.__labels:
                outfile = outdir/f'{label}.pdf'
                pdffiles[label] = PdfPages(outfile)

            for window_x, window_y, info in zip(s.x, s.y, s.info):
                label = np.bincount(window_y).argmax()

                for i in range(window_x.shape[-1]):
                    p = plt.plot(window_x[:,i])
                    for x, y, a in zip(range(window_x.shape[0]), window_x[:,i], window_y):
                        plt.annotate(a, (x, y))

                plt.suptitle('\n'.join([' '.join(inf) for inf in np.unique(info, axis=0)]))

                pdffiles[label].savefig(fig)
                fig.clf()

            for pdffile in pdffiles.values():
                pdffile.close()
        return datamodel
