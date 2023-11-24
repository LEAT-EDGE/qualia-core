import logging
import time

import numpy as np

from .Preprocessing import Preprocessing

logger = logging.getLogger(__name__)

class MFCC(Preprocessing):
    def __init__(self,
                 sample_rate: int,
                 n_mfcc: int,
                 dct_type: int=2,
                 norm: str='ortho',
                 log_mels=False,
                 melkwargs: dict={},
                 chunks: int = 4,
                 dims: int = 1):
        import torchaudio
        self.mfcc_transform = torchaudio.transforms.MFCC(sample_rate=sample_rate,
                                                        n_mfcc=n_mfcc,
                                                        dct_type=dct_type,
                                                        norm=norm,
                                                        log_mels=log_mels,
                                                        melkwargs=melkwargs)
        self.__chunks = chunks
        self.__dims = dims

    def __call__(self, datamodel):
        start = time.time()
        import torch
        for name, s in datamodel:
            # 1D
            s.x = s.x.squeeze(-1)
            s.x = torch.tensor(s.x)
            chunks = torch.chunk(s.x, self.__chunks)
            chunks_mfcc = [self.mfcc_transform(chunk) for chunk in chunks]
            s.x = torch.concat(chunks_mfcc)
            s.x = s.x.numpy()
            s.x = s.x.swapaxes(1, 2)
            if self.__dims == 2:
                s.x = np.expand_dims(s.x, -1)

            # 2D, add channels dim
            #s.x = self.mfcc_transform(torch.tensor(s.x.squeeze(-1))).unsqueeze(-1).numpy()

        logger.info('Elapsed: %s s', time.time() - start)
        return datamodel
