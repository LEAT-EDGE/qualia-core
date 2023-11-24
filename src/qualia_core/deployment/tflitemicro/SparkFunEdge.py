from pathlib import Path
import shutil
import subprocess
import sys
import os
from collections import namedtuple
from importlib.resources import files

from .TFLite2CArray import TFLite2CArray

class SparkFunEdge:
    import qualia_core.evaluation.target.Qualia as evaluator # Suggested evaluator

    def __init__(self,
        dev: str='/dev/ttyUSB0',
        baudrate: int=921600,
        projectdir: Path=files('qualia_core.assets')/'projects'/'tflitemicro'/'SparkFunEdge',
        tfpath: Path=Path('third_party/tflite-micro'),
        outdir: Path=Path('out/deploy/sparkfun_edge')):
        self.__dev = dev
        self.__projectdir = projectdir
        self.__projectdir_from_tfpath = Path(os.path.relpath(projectdir.resolve(), tfpath.resolve()))
        self.__tfpath = tfpath
        self.__baudrate = baudrate
        self.__outdir = outdir
        self.__numthreads = os.cpu_count()

    def __find_ambiqsuite(self, tfpath):
        return sorted((tfpath/'tensorflow'/'lite'/'micro'/'tools'/'make'/'downloads').glob('AmbiqSuite-*'), reverse=True)[0]

    def __create_outdir(self, outdir: Path):
        outdir.mkdir(parents=True, exist_ok=True)

    def __write_model(self, model, modelpath):
        with modelpath.open('w') as f:
            f.write(TFLite2CArray().convert(model.data, model.input_shape).cc)
    
    def __clean(self, tfpath):
        cmd = ['make', '-C', str(tfpath),
                         '-f', 'tensorflow/lite/micro/tools/make/Makefile',
                         '-f', str(self.__projectdir_from_tfpath/'Makefile.inc'),
                         'TARGET=sparkfun_edge',
                         'TARGET_ARCH=cortex-m4',
                         'clean']
        print(cmd)
        s = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr, text=True)
        return s.returncode == 0

    def __third_party_downloads(self, tfpath, cmsis):
        cmd = ['make', '-C', str(tfpath),
                         '-j', '1',
                         '-f', 'tensorflow/lite/micro/tools/make/Makefile',
                         '-f', str(self.__projectdir_from_tfpath/'Makefile.inc'),
                         'TARGET=sparkfun_edge',
                         'TARGET_ARCH=cortex-m4',
                         'third_party_downloads']
        if cmsis:
            cmd.append('OPTIMIZED_KERNEL_DIR=cmsis_nn')
        print(cmd)
        s = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr, text=True)
        return s.returncode == 0

    def __build(self, tfpath, cmsis):
        cmd = ['make', '-C', str(tfpath),
                         '-j', str(self.__numthreads),
                         '-f', 'tensorflow/lite/micro/tools/make/Makefile',
                         '-f', str(self.__projectdir_from_tfpath/'Makefile.inc'),
                         'TARGET=sparkfun_edge',
                         'TARGET_ARCH=cortex-m4',
                         'myapp_bin']
        if cmsis:
            cmd.append('OPTIMIZED_KERNEL_DIR=cmsis_nn')
        print(cmd)
        s = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr, text=True)
        return s.returncode == 0

    def __sign(self, tfpath, outdir, tag: str):
        # use default  keys
        d = self.__find_ambiqsuite(self.__tfpath)/'tools'/'apollo3_scripts'
        shutil.copy(d/'keys_info0.py', d/'keys_info.py')

        s = subprocess.run(['python3',
            str(self.__find_ambiqsuite(self.__tfpath)/'tools'/'apollo3_scripts'/'create_cust_image_blob.py'),
            '--bin', str(tfpath/'tensorflow'/'lite'/'micro'/'tools'/'make'/'gen'/'sparkfun_edge_cortex-m4_default'/'bin'/'myapp.bin'),
            '--load-address', '0xC000', '--magic-num', '0xCB', '-o', str(outdir/f'{tag}_nonsecure_ota'), '--version', '0x0'],
            stdout=sys.stdout, stderr=sys.stderr, text=True)
        return s.returncode == 0

    def __create_image(self, tfpath, outdir, tag: str):
        s = subprocess.run(['python3',
            str(self.__find_ambiqsuite(self.__tfpath)/'tools'/'apollo3_scripts'/'create_cust_wireupdate_blob.py'),
            '--load-address', '0x20000', '--bin', str(outdir/f'{tag}_nonsecure_ota.bin'), '-i', '6', '-o', str(outdir/f'{tag}_nonsecure_wire'),
            '--options', '0x1'],
            stdout=sys.stdout, stderr=sys.stderr, text=True)
        return s.returncode == 0

    def __upload(self, dev: str, baudrate: int, tfpath, outdir, tag: str):
        s = subprocess.run(['python3',
            str(self.__find_ambiqsuite(self.__tfpath)/'tools'/'apollo3_scripts'/'uart_wired_update.py'),
            '-b', str(baudrate), dev, '-r', '1', '-f', str(outdir/f'{tag}_nonsecure_wire.bin'), '-i', '6'],
            stdout=sys.stdout, stderr=sys.stderr, text=True)
        return s.returncode == 0

    def prepare(self, tag, model, optimize=None, compression: int=None):
        if compression != 1:
            raise ValueError(f'Compression not supported by {self.__class__.__name__}')

        if not optimize:
            cmsis = False
        elif optimize == 'cmsis-nn':
            cmsis = True # Use CMSIS-NN kernel in TFLiteMicro
            print(f'FIXME: Evaluation not logging optimization')
        else:
            raise ValueError(f'Unsupported {optimize} optimization by {self.__class__.__name__}')

        self.__create_outdir(outdir=self.__outdir)

        if not self.__clean(tfpath=self.__tfpath):
            return None
        self.__write_model(model, modelpath=self.__projectdir/'model_data.cc')
        if not self.__third_party_downloads(tfpath=self.__tfpath, cmsis=cmsis):
                return None
        if not self.__build(tfpath=self.__tfpath, cmsis=cmsis):
            return None
        if not self.__sign(tfpath=self.__tfpath, outdir=self.__outdir, tag=tag):
            return None
        if not self.__create_image(tfpath=self.__tfpath, outdir=self.__outdir, tag=tag):
            return None
        return self

    def deploy(self, tag):
        input('Put target in programming mode and press Enter…')
        if not self.__upload(dev=self.__dev, baudrate=self.__baudrate, tfpath=self.__tfpath, outdir=self.__outdir, tag=tag):
            return None
        return namedtuple('Deploy', ['rom_size', 'ram_size', 'evaluator'])(self.__rom_size(tag), None, self.evaluator)

    def __rom_size(self, tag):
        return (self.__outdir/f'{tag}_nonsecure_wire.bin').stat().st_size
