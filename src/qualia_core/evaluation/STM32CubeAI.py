from pathlib import Path
import re
import sys
from qualia_core.utils.process import subprocesstee                                                                                     
from qualia_core.evaluation.Stats import Stats

class STM32CubeAI:
    def __init__(self,
        stm32cubeai_args=tuple(),
        mode=''):
        self.__mode = mode
        self.__stm32cubeai_args = stm32cubeai_args

        # Project not compatible with 6.0.0 yet, and 6.0.0 is buggy anyway. Use 5.2.0
        self.__stm32cubeai_bin = Path.home()/'STM32Cube'/'Repository'/'Packs'/'STMicroelectronics'/'X-CUBE-AI'/'5.2.0'/'Utilities'/'linux'/'stm32ai'
        #self.__stm32cubeai_bin = next((Path.home()/'STM32Cube'/'Repository'/'Packs'/'STMicroelectronics'/'X-CUBE-AI').glob('*'))/'Utilities'/'linux'/'stm32ai'

    def __dataset_to_csv(self, dataset, outdir, limit):
        import numpy as np
        testX = dataset.sets.test.x
        testY = dataset.sets.test.y
        if limit:
            testX = testX[:limit]
            testY = testY[:limit]
        testX = testX.reshape((testX.shape[0], -1))
        testY = testY.reshape((testY.shape[0], -1))
        np.savetxt(outdir/'testX.csv', testX, delimiter=",", fmt='%f')
        np.savetxt(outdir/'testY.csv', testY, delimiter=",", fmt='%f')

    def __validate(self, testX: Path, testY: Path, modelpath: Path, compression: int, outdir: Path, logdir: Path, tag: str):

        cmd = str(self.__stm32cubeai_bin)
        args =  ('validate',
                    '--name', 'network',
                    '--model', str(modelpath),
                    '--mode', self.__mode,
                    #'--type', 'keras', # automatically detected, using tflite for now
                    #'--compression', str(compression),
                    '--valinput', str(testX),
                    '--valoutput', str(testY),
                    '--verbosity', '1',
                    '--workspace', str(outdir/'workspace'),
                    '--output', str(outdir/tag),
                    '--classifier'
                ) + self.__stm32cubeai_args
        print(cmd, *args)
        with (logdir/f'{tag}.txt').open('wb') as logfile:
            logfile.write(' '.join([str(cmd), *args, '\n']).encode('utf-8'))
            returncode, outputs = subprocesstee.run(cmd, *args, files={sys.stdout: logfile, sys.stderr: logfile})
        return returncode, outputs

    def __parse_validate_stdout(self, s: str):
        duration = re.search("^\s*duration\s*:\s*([.\d]+)\s+ms\s+\(average\)$", s, re.MULTILINE)
        if duration is not None:
            duration = float(duration.group(1))/1000

        accuracy = re.search(f'^{self.__mode}\ C-model\ #1\s+([.\d]+)%.*$', s, re.MULTILINE)
        if accuracy is not None:
            accuracy = float(accuracy.group(1))/100

        return duration, accuracy

    def evaluate(self, framework, model_kind, dataset, target: str, tag: str, limit: int=None, dataaugmentations=[]):
        if dataaugmentations:
            raise ValueError(f'dataaugmentations not supported for {self.__class__.__name__}')

        outdir = Path('out')/'deploy'/target
        logdir = Path('out')/'evaluate'/target
        logdir.mkdir(parents=True, exist_ok=True)

        self.__dataset_to_csv(dataset, logdir, limit)

        return_code, outputs = self.__validate(testX=logdir/'testX.csv',
                                                testY=logdir/'testY.csv',
                                                modelpath=Path('out')/'deploy'/'stm32cubeai'/f'{tag}.tflite',
                                                compression=None,
                                                outdir=outdir,
                                                logdir=logdir,
                                                tag=tag)
        if return_code != 0:
            return None
        duration, accuracy = self.__parse_validate_stdout(outputs[1].decode())

        return Stats(avg_time=duration, accuracy=accuracy)
