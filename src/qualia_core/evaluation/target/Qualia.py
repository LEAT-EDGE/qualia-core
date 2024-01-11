from __future__ import annotations

import logging
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple

from qualia_core.evaluation.Evaluator import Evaluator
from qualia_core.evaluation.Stats import Stats
from qualia_core.typing import TYPE_CHECKING
from qualia_core.utils.logger.CSVLogger import CSVLogger

# We are inside a TYPE_CHECKING block but our custom TYPE_CHECKING constant triggers TCH001-TCH003 so ignore them
if TYPE_CHECKING:
    from qualia_core.dataaugmentation.DataAugmentation import DataAugmentation  # noqa: TCH001
    from qualia_core.datamodel.RawDataModel import RawDataModel  # noqa: TCH001
    from qualia_core.learningframework.LearningFramework import LearningFramework  # noqa: TCH001

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)

class Result(NamedTuple):
    i: int = -1
    y: int = -1
    score: float = -1
    time: float = -1


class Qualia(Evaluator):
    """Custom evaluation loop for Qualia embedded implementations like TFLite Micro and Qualia-CodeGen."""

    def __init__(self,
                 dev: str = '/dev/ttyUSB0',
                 baudrate: int = 921600,
                 timeout: int = 30, # 30 seconds default transmission timeout
                 shuffle: bool = False) -> None:  # noqa: FBT001, FBT002
        super().__init__()

        self.__dev = Path(dev)
        self.__baudrate = baudrate
        self.__shuffle = shuffle
        self.__timeout = timeout

    def __get_dev(self, path: Path) -> Path | None:
        dev: Path | None = None
        logger.info('Waiting for device "%s"…', path)
        start_time = time.time()
        while dev is None:
            if time.time() - start_time > self.__timeout:
                logger.error('Timeout looking up for device "%s"', path)
                break
            devs = list(path.parent.glob(path.name))
            if devs:
                if len(devs) > 1:
                    logger.warning('%d devices matched, using first device %s', len(devs), devs[0])
                dev = devs[0]
            time.sleep(0.1)
        return dev

    @override
    def evaluate(self,
                 framework: LearningFramework[Any],
                 model_kind: str,
                 dataset: RawDataModel,
                 target: str,
                 tag: str,
                 limit: int | None = None,
                 dataaugmentations: list[DataAugmentation] | None = None) -> Stats | None:
        import serial

        dev = self.__get_dev(self.__dev)
        if dev is None:
            logger.error('No device found.')
            return None

        if dataset.sets.test is None:
            logger.error('Test dataset is required')
            raise ValueError

        test_x = dataset.sets.test.x.copy()
        test_y = dataset.sets.test.y.copy()

        # Shuffle
        if self.__shuffle:
            test_x, test_y = self.shuffle_dataset(test_x, test_y)

        test_x, test_y = self.apply_dataaugmentations(framework, dataaugmentations, test_x, test_y)

        test_x, test_y = self.limit_dataset(test_x, test_y, limit)

        logger.info('Reset the target…')
        s = serial.Serial(str(dev), self.__baudrate, timeout=self.__timeout)
        logger.info('Device: %s', s.name)
        r = s.readline().decode('cp437')

        start_time = time.time()
        logger.info('Waiting for READY from device "%s"…', dev)
        while 'READY' not in r:
            if time.time() - start_time > self.__timeout:
                logger.error('Timeout waiting for READY for device "%s"', dev)
                return None

            r = s.readline().decode('cp437')
            time.sleep(0.1)

        # create log directory
        (Path('logs')/'evaluate'/target).mkdir(parents=True, exist_ok=True)

        results: list[Result] = [] # read from target

        log: CSVLogger[Result] = CSVLogger(name=__name__,
                        file=Path('evaluate')/target/f'{tag}_{datetime.now():%Y-%m-%d_%H-%M-%S}.txt')  # noqa: DTZ005 system tz ok
        log.fields = Result

        for i, line in enumerate(test_x):
            msg = ','.join(map(str, line.flatten())) + '\r\n'
            _ = s.write(msg.encode('cp437')) # Send test vector

            r = s.readline() # Read acknowledge
            tstart = time.time() # Start timer
            r = r.decode('cp437')
            if not r or int(r) != len(msg): # Timed out or didn't receive all the data
                logger.error('%d: Transmission error: %s != %d', i, int(r), len(msg))
                return None

            r = s.readline() # Read result
            tstop = time.time() # Stop timer
            r = r.decode('cp437').rstrip().split(',')
            r = Result(i=int(r[0]),
                       y=int(r[1]),
                       score=float(r[2]),
                       time=tstop-tstart)
            logger.info('%d: %s', i, str(r))
            results.append(r)

            # Log result to file
            log(r)

        avg_time = statistics.mean([r.time for r in results])

        return Stats(avg_time=avg_time, accuracy=self.compute_accuracy(preds=[r.y for r in results], truth=test_y))

        # avg it/secs
        # ram usage
        # rom usage
        # cpu type
        # cpu model
        # cpu freq
        # accuracy
        # power consumption
