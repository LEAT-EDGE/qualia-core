from __future__ import annotations

import logging
import sys

import numpy as np
import numpy.typing
import torch

from qualia_core.typing import TYPE_CHECKING

from .DataAugmentationPyTorch import DataAugmentationPyTorch

if TYPE_CHECKING:
    import torch  # noqa: TCH002

if sys.version_info >= (3, 12):
    from typing import TypedDict, override
else:
    from typing_extensions import TypedDict, override

logger = logging.getLogger(__name__)

class MelConfigDict(TypedDict):
    n_fft: int
    n_mels: int
    f_min: int
    f_max: int

class CMSISMFCC(DataAugmentationPyTorch):
    def __init__(self,  # noqa: PLR0913
                 sample_rate: int,
                 n_mfcc: int,
                 melkwargs: MelConfigDict,
                 evaluate: bool = False,  # noqa: FBT001, FBT002
                 before: bool = False,  # noqa: FBT001, FBT002
                 after: bool = True) -> None:  # noqa: FBT001, FBT002
        import cmsisdsp  # type: ignore[import-untyped]
        import cmsisdsp.mfcc  # type: ignore[import-untyped]

        super().__init__(evaluate=evaluate, before=before, after=after)

        self.__n_mfcc = n_mfcc
        self.__n_fft = melkwargs['n_fft']

        # Numpy Hann window
        #window = np.hanning(melkwargs['n_fft'])
        # Torch Hann window
        window = torch.hann_window(melkwargs['n_fft'])

        # CMSIS-DSP Mel filterbank generation
        '''
        len_filt: list[int]
        pos_filt: list[int]
        filt_packed: numpy.typing.NDArray[np.float32]
        len_filt, pos_filt, filt_packed = cmsisdsp.mfcc.melFilterMatrix(cmsisdsp.datatype.F32,
                                                                      melkwargs['f_min'],
                                                                      melkwargs['f_max'],
                                                                      melkwargs['n_mels'],
                                                                      sample_rate,
                                                                      melkwargs['n_fft'])
        '''
        # TorchAudio Mel filterbank generation
        import torchaudio
        mel_fb = torchaudio.functional.melscale_fbanks(melkwargs['n_fft'] // 2 + 1,
                                                       melkwargs['f_min'],
                                                       melkwargs['f_max'],
                                                       melkwargs['n_mels'],
                                                       sample_rate,
                                                       norm=None,
                                                       mel_scale='htk').transpose(-1, -2).numpy()
        len_filt, pos_filt, filt_packed = self.__sparse_matrix(mel_fb)

        # CMSIS-DSP DCT-II orthogonal matrix generation
        #filt_dct: numpy.typing.NDArray[np.float32] = cmsisdsp.mfcc.dctMatrix(cmsisdsp.datatype.F32, n_mfcc, melkwargs['n_mels'])
        #filt_dct[0:melkwargs['n_mels']] /= np.sqrt(2.0) # Use same normalization as torchaudio

        # TorchAudio DCT-II orthogonal matrix generation
        import torchaudio
        filt_dct = torchaudio.functional.create_dct(n_mfcc, melkwargs['n_mels'], norm='ortho').transpose(-1, -2).numpy()

        self.__mfccf32 = cmsisdsp.arm_mfcc_instance_f32()

        status = cmsisdsp.arm_mfcc_init_f32(
            self.__mfccf32,
            melkwargs['n_fft'],
            melkwargs['n_mels'],
            n_mfcc,
            filt_dct,
            pos_filt,
            len_filt,
            filt_packed,
            window)
        if status != 0:
            logger.error('arm_mfcc_init_f32() error: %d', status)
            raise RuntimeError

    # From CMSIS-DSP cmsisdsp/mfcc.py melFilterMatrix()
    def __sparse_matrix(self, mel_fb: numpy.typing.NDArray[np.float32]) -> tuple[list[int],
                                                                                 list[int],
                                                                                 numpy.typing.NDArray[np.float32]]:
        pos_filt: list[int] = []
        len_filt: list[int] = []
        total_len = 0
        filt_packed: list[float] = []
        start_pos = 0
        end_pos = 0
        for n in range(mel_fb.shape[0]):
            nb = 0
            start_found = False
            for sample in mel_fb[n, :]:
                if not start_found and sample != 0.0:
                    start_found = True
                    start_pos = nb

                if start_found and sample == 0.0:
                    end_pos = nb - 1
                    break
                nb = nb + 1
            len_filt.append(end_pos - start_pos + 1)
            total_len += end_pos - start_pos + 1
            pos_filt.append(start_pos)
            filt_packed += list(mel_fb[n, start_pos:end_pos+1])
        return len_filt, pos_filt, np.array(filt_packed, dtype=np.float32)

    def apply(self, x: torch.Tensor, y: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        import cmsisdsp  # type: ignore[import-untyped]

        output_shape_channels_last = (x.shape[0], x.shape[-1] // self.__n_fft, x.shape[1], self.__n_mfcc)

        # Drop extra samples that cannot be covered by a full window
        max_length = (x.shape[2] // self.__n_fft) * self.__n_fft
        x_truncated = x[:,:,:max_length]

        x_windowed = x_truncated.reshape((x.shape[0], x.shape[1], x.shape[2] // self.__n_fft, self.__n_fft))
        x_windowed_cpu = x_windowed.cpu().numpy() # Need to process on CPU to call CMSIS-DSP

        x_mfcc = np.zeros(output_shape_channels_last)

        tmp = np.zeros(self.__n_fft + 2)

        for i, input_vector in enumerate(x_windowed_cpu):
            for j, channel in enumerate(input_vector):
                for k, window in enumerate(channel):
                    x_mfcc[i][k][j] = cmsisdsp.arm_mfcc_f32(self.__mfccf32,
                                                            window,
                                                            tmp)

        x_mfcc_concat_channels = x_mfcc.reshape(x_mfcc.shape[0], x_mfcc.shape[1], x_mfcc.shape[2] * x_mfcc.shape[3])
        x_mfcc_channels_first = x_mfcc_concat_channels.swapaxes(1, 2)

        return torch.tensor(x_mfcc_channels_first, dtype=x.dtype, device=x.device), y

    @override
    def __call__(self, data: tuple[torch.Tensor, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        return self.apply(*data, device=device)
