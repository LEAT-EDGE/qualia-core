from string import Template
from pathlib import Path

class TFLite2CArray:
    def __init__(self):
        with (Path(__file__).parent.parent.parent/'assets'/'template'/'tflite_model_data.cc').open('r') as f:
            self.__templatecc = Template(f.read())

    def convert(self, data, input_shape):
        datax = [hex(b) for b in data]
        self.__output = ', '.join(datax)
        self.__datalen = len(datax)
        self.__dims = ', '.join((str(v) if v != None else '1' for v in input_shape)) # convert Tuple to string replacing None by 1
        self.__dimslen = len(input_shape)
        return self

    @property
    def cc(self):
        return self.__templatecc.substitute(data=self.__output, datalen=self.__datalen, dims=self.__dims, dimslen=self.__dimslen)
