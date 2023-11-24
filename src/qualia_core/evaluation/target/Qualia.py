from datetime import datetime
from pathlib import Path
from collections import namedtuple
import sys
import csv
import time
import statistics
from glob import glob
from qualia_core.evaluation.Stats import Stats

class Qualia:
    '''Custom evaluation loop for Qualia embedded implementations like TFLite Micro and Qualia-CodeGen'''
    def __init__(self,
        dev: str='/dev/ttyUSB0',
        baudrate: int=921600):

        self.__dev = dev

        self.__baudrate = baudrate
        self.__timeout = 30 # 30 seconds transmission timeout

    def __get_dev(self):
        dev = None
        print(f'Waiting for device "{self.__dev}"…')
        start_time = time.time()
        while dev is None:
            if time.time() - start_time > self.__timeout:
                print('Timeout looking up for device "{self.__dev}"', file=sys.stderr)
                break
            devs = glob(self.__dev)
            if devs:
                if len(devs) > 1:
                    print(f'Warning: {len(devs)} devices matched, using first device {devs[0]}', file=sys.stderr)
                dev = devs[0]
            time.sleep(0.1)
        return dev

    def evaluate(self, framework, model_kind, dataset, target: str, tag: str, limit: int=None, dataaugmentations=[]):
        import serial

        dev = self.__get_dev()

        testX = dataset.sets.test.x.copy()
        testY = dataset.sets.test.y.copy()
        # Apply evaluation "dataaugmentations" to dataset
        for da in dataaugmentations:
            if da.evaluate:
                testX, testY = framework.apply_dataaugmentation(da, testX, testY)

        print('Reset the target…')
        s = serial.Serial(dev, self.__baudrate, timeout=self.__timeout)
        print(s.name)
        r = s.readline().decode('cp437')

        start_time = time.time()
        print(f'Waiting for READY from device "{dev}"…')
        while not 'READY' in r:
            if time.time() - start_time > self.__timeout:
                print('Timeout waiting for READY for device "{self.__dev}"', file=sys.stderr)
                break
            r = s.readline().decode('cp437')
            time.sleep(0.1)



        # create log directory
        (Path('logs')/'evaluate'/target).mkdir(parents=True, exist_ok=True)

        Result = namedtuple('Result', ['i', 'y', 'score', 'time'], defaults=[-1, -1, -1, -1])
        results = [] # read from target
        with (Path('logs')/'evaluate'/target/f'{tag}_{datetime.now():%Y-%m-%d_%H-%M-%S}.txt').open('w', newline='') as logfile:
            logwriter = csv.writer(logfile)
            logwriter.writerow(Result._fields)

            for i, line in enumerate(testX):
                print(f'{i}: ', end='')
                line = ','.join(map(str, line.flatten())) + '\r\n'
                s.write(line.encode('cp437')) # Send test vector

                r = s.readline() # Read acknowledge
                tstart = time.time() # Start timer
                r = r.decode('cp437')
                if not r or int(r) != len(line): # Timed out or didn't receive all the data
                    print(f'Transmission error: {r} != {len(line)}')
                    return None

                r = s.readline() # Read result
                tstop = time.time() # Stop timer
                r = r.decode('cp437').rstrip().split(',')
                r = Result(*r, time=tstop-tstart)
                print(r)
                results.append(r)

                # Log result to file
                logwriter.writerow(r)

                # Only infer 'limit' vectors
                if limit is not None and i + 1 >= limit:
                    break

        avg_time = statistics.mean([r.time for r in results])

        # Compute accuracy
        correct = 0
        for line,result in zip(testY, results):
            if line.argmax() == int(result.y):
                correct += 1

        accuracy = correct / len(results)

        return Stats(avg_time=avg_time, accuracy=accuracy)

        # avg it/secs
        # ram usage
        # rom usage
        # cpu type
        # cpu model
        # cpu freq
        # accuracy
        # power consumption
