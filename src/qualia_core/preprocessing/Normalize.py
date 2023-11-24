from .Preprocessing import Preprocessing

class Normalize(Preprocessing):
    def z_score(self, data):
        if self.debug:
            print(f'Before normalization: train mean={data.sets.train.x.mean()}, train std={data.sets.train.x.std()}, test mean={data.sets.test.x.mean()}, test std={data.sets.test.x.std()}')

        x_mean = data.sets.train.x.mean(axis=self.__axis, keepdims=True)
        x_std = data.sets.train.x.std(axis=self.__axis, keepdims=True)

        for _, s in data:
            s.x -= x_mean
            s.x /= x_std

        if self.debug:
            print(f'After normalization: train mean={data.sets.train.x.mean()}, train std={data.sets.train.x.std()}, test mean={data.sets.test.x.mean()}, test std={data.sets.test.x.std()}')

        return data

    def min_max(self, data):
        if self.debug:
            print(f'Before normalization: train mean={data.sets.train.x.mean()}, train std={data.sets.train.x.std()}, test mean={data.sets.test.x.mean()}, test std={data.sets.test.x.std()}')
            print(f'                      train min={data.sets.train.x.min()}, train max={data.sets.train.x.max()}, test min={data.sets.test.x.min()}, test max={data.sets.test.x.max()}')

        x_min = data.sets.train.x.min(axis=tuple(self.__axis), keepdims=True)
        x_max = data.sets.train.x.max(axis=tuple(self.__axis), keepdims=True)

        for _, s in data:
            s.x -= x_min
            s.x /= (x_max - x_min)

        if self.debug:
            print(f'After normalization: train mean={data.sets.train.x.mean()}, train std={data.sets.train.x.std()}, test mean={data.sets.test.x.mean()}, test std={data.sets.test.x.std()}')
            print(f'                     train min={data.sets.train.x.min()}, train max={data.sets.train.x.max()}, test min={data.sets.test.x.min()}, test max={data.sets.test.x.max()}')

        return data

    methods = {
        'z-score': z_score,
        'min-max': min_max
    }

    def __init__(self, method: str='z-score', axis: tuple = (0,), debug: bool=False):
        self.__refdata = None
        self.debug = debug
        if method not in self.methods:
            raise ValueError(f'{self.__class__.__name__}: Method {method} is not supported. Supported methods: {", ".join(self.methods)}')
        else:
            self.__method = self.methods[method].__get__(self)
        self.__axis = axis

    def __call__(self, *args, **kwargs):
        return self.__method(*args, **kwargs)
