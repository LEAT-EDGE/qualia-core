from __future__ import annotations

import functools
import logging
import math
import multiprocessing
import os
import pickle
import sys
import time
from collections import OrderedDict
from collections.abc import Generator
from concurrent.futures import Future, ProcessPoolExecutor
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path
from typing import Any, cast

import blosc2  # type: ignore[import-untyped]
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing
import torch
from matplotlib.backends.backend_pdf import PdfPages
from torch import nn

from qualia_core.datamodel.RawDataModel import RawData
from qualia_core.learningframework.PyTorch import PyTorch
from qualia_core.learningmodel.pytorch.Quantizer import Quantizer
from qualia_core.postprocessing.PostProcessing import PostProcessing
from qualia_core.typing import TYPE_CHECKING, ModelConfigDict

if TYPE_CHECKING:
    from multiprocessing.shared_memory import SharedMemory  # noqa: TCH003

    from matplotlib.image import AxesImage  # noqa: TCH002

    from qualia_core.qualia import TrainResult  # noqa: TCH001

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)

class VisualizeFeatureMaps(PostProcessing[nn.Module]):
    figsize: tuple[float, float] = (11.7, 8.3) # A4 landscape in inches
    outdir: Path = Path('out')/'feature_maps'

    def __init__(self,
                 create_pdf: bool = True,  # noqa: FBT001, FBT002
                 save_feature_maps: bool = True,  # noqa: FBT001, FBT002
                 compress_feature_maps: bool = True,  # noqa: FBT001, FBT002
                 data_range: list[int] | None = None) -> None:
        super().__init__()
        self.__create_pdf = create_pdf
        self.__save_feature_maps = save_feature_maps
        self.__compress_feature_maps = compress_feature_maps
        if data_range is not None and len(data_range) < 2:  # noqa: PLR2004
            logger.error('data_range must be a list of 2 integers')
            raise ValueError
        self.__data_range = (data_range[0], data_range[1]) if data_range is not None else (None, None)

    def __gen_featuremaps(self,
                          trainresult: TrainResult,
                          data_range: tuple[int | None, int | None]) -> OrderedDict[str, numpy.typing.NDArray[Any]] | None:
        feature_maps: OrderedDict[str, torch.Tensor] = OrderedDict()
        framework = trainresult.framework
        model = trainresult.model
        dataset = trainresult.testset

        # Extract range from dataset
        dataset_cut = RawData(x=dataset.x[data_range[0]:data_range[1]],
                              y=dataset.y[data_range[0]:data_range[1]])


        if not isinstance(framework, PyTorch):
            logger.error('Only compatible with PyTorch-based frameworks')
            return None

        def hook(layername: str, _: nn.Module, x: torch.Tensor, output: torch.Tensor) -> None:
            out_sum = output.detach().sum(0) # Sum over batch dimension

            # Accumulate for each new batch
            if layername not in feature_maps:
                feature_maps[layername] = out_sum.clone()
            else:
                feature_maps[layername] += out_sum


        handles = [layer.register_forward_hook(functools.partial(hook,
                                                                 layername))
                   for layername, layer in cast(Generator[tuple[str, nn.Module], None, None], model.named_modules())
                   if not isinstance(layer, Quantizer)] # No hook to register for Quantizer module

        _ = framework.predict(model=model,
                              dataset=dataset_cut,
                              batch_size=trainresult.batch_size,
                              dataaugmentations=trainresult.dataaugmentations,
                              experimenttracking=None,
                              name=trainresult.name)

        for handle in handles:
            handle.remove()

        # Make sure everything is on CPU first
        feature_maps_numpy = OrderedDict((layername, feature_map.cpu().numpy()) for layername, feature_map in feature_maps.items())

        # Add input image, transposed to channels_last to be the same as PyTorch feature_maps
        input_data = dataset_cut.x.mean(0).astype(dataset_cut.x.dtype)
        if len(input_data.shape) == 3:
            feature_maps_numpy['__INPUT__'] = input_data.transpose(2, 0, 1)
        elif len(input_data.shape) == 2:
            feature_maps_numpy['__INPUT__'] = input_data.swapaxes(0, 1)
        else:
            feature_maps_numpy['__INPUT__'] = input_data

        feature_maps_numpy.move_to_end('__INPUT__', last=False) # Move __INPUT__ to beginning of OrderedDict

        return feature_maps_numpy

    @staticmethod
    def _init_process() -> None:
        # Need to configure root logger since we use spawn instead of fork
        from qualia_core.utils.logger.setup_root_logger import setup_root_logger

        setup_root_logger(colored=True)

    @staticmethod
    def _gen_pdf_channel_page(feature_map_chunk: numpy.typing.NDArray[Any],  # noqa: PLR0913
                              feature_map_chunk_start_index: int,
                              pdf: PdfPages,
                              layername: str,
                              im: AxesImage,
                              vmin: float,
                              vmax: float,
                              colormap: str,
                              figsize: tuple[float, float]) -> None:
        ncols = math.ceil(math.sqrt(len(feature_map_chunk)) * 1.2) # Rectangle grid with ratio ~= 6/5
        nrows = math.ceil(len(feature_map_chunk) / ncols)
        fig, axs = plt.subplots(nrows,
                                ncols,
                                sharex='all', # Can cause noticeable slowdowns with large number of plots
                                sharey='all', # Can cause noticeable slowdowns with large number of plots
                                squeeze=False,
                                figsize=figsize,
                                layout='compressed')
        _ = fig.suptitle(layername, fontsize=8)
        for i, (ax, channel) in enumerate(zip(axs.flatten(), feature_map_chunk)):
            ax.set_title(str(feature_map_chunk_start_index + i))
            ax.imshow(channel,  interpolation='nearest', vmin=vmin, vmax=vmax, cmap=colormap)

        # Turn off unused subplots
        for ax in axs.flatten()[len(feature_map_chunk):]:
            ax.set_axis_off()

        # Use same colorbar reference as the layer plot
        cb = fig.colorbar(im, ax=axs, use_gridspec=True, orientation='vertical', label='Total activity')
        cb.mappable.set_clim(vmin, vmax)
        pdf.savefig()
        plt.close()

    @staticmethod
    def _gen_pdf_input_layer_rgb(data: numpy.typing.NDArray[Any],
                                 figsize: tuple[float, float],
                                 pdf: PdfPages) -> None:
        _ = plt.figure(figsize=figsize)
        _ = plt.title('Input image')
        _ = plt.xlabel('Width')
        _ = plt.ylabel('Height')
        _ = plt.imshow(data.transpose(1, 2, 0), interpolation='nearest') # channels_first in PyTorch, channels_last for matplotlib
        pdf.savefig()
        plt.close()


    @staticmethod
    def _compute_scale(data: numpy.typing.NDArray[Any]) -> tuple[float, float]:
        # Symmetric scale
        vmax = np.abs(data).max().item()
        # Prevent collapsing of scale
        if vmax == 0.0:
            vmax = 1.0
        vmin = -vmax
        return vmin, vmax

    @staticmethod
    def _gen_pdf_layer(layername: str,  # noqa: PLR0913
                       shared_buf: SharedMemory,
                       shared_buf_dtype: np.dtype[Any],
                       shared_buf_shape: tuple[int, ...],
                       outfile_pdf: Path,
                       figsize: tuple[float, float]) -> Path:
        start = time.time()

        logger.info("%s generating PDF for layer '%s'…", multiprocessing.current_process().name, layername)

        feature_map = np.frombuffer(shared_buf.buf, dtype=shared_buf_dtype)
        feature_map = feature_map.reshape(shared_buf_shape)

        plt.rcParams.update({'font.size': 8})

        with PdfPages(outfile_pdf) as pdf:
            if len(feature_map.shape) < 2: # Flattened data is expanded to 1D + channels  # noqa: PLR2004
                feature_map = np.expand_dims(feature_map, 0)
            if len(feature_map.shape) < 3: # 1D + channels data is expanded to 2D + channels  # noqa: PLR2004
                feature_map = np.expand_dims(feature_map, 1)

            # If input tensor and 3 channels, display image as RGB
            if layername == '__INPUT__' and len(feature_map) == 3: # Assume RGB input  # noqa: PLR2004
                VisualizeFeatureMaps._gen_pdf_input_layer_rgb(data=feature_map, figsize=figsize, pdf=pdf)

            # Scale for sum of channels
            vmin, vmax = VisualizeFeatureMaps._compute_scale(feature_map.sum(0))
            # Use a diverging colormap
            colormap = 'seismic'

            fig = plt.figure(figsize=figsize)
            _ = plt.title(layername)
            _ = plt.xlabel('Width')
            _ = plt.ylabel('Height')
            im = plt.imshow(feature_map.sum(0), interpolation='nearest', vmin=vmin, vmax=vmax, cmap=colormap)
            cb = fig.colorbar(im, orientation='vertical', label='Total activity')
            cb.mappable.set_clim(vmin, vmax)
            pdf.savefig()
            plt.close()

            # Page(s) for per-channel plots
            subplot_pages = math.ceil(len(feature_map) / 70) # Max. 70 channel per page
            feature_map_chunks = np.array_split(feature_map, subplot_pages)

            # Scale for channels
            vmin, vmax = VisualizeFeatureMaps._compute_scale(feature_map)

            i = 0 # Feature map index for title
            for feature_map_chunk in feature_map_chunks:
                VisualizeFeatureMaps._gen_pdf_channel_page(feature_map_chunk=feature_map_chunk,
                                                           feature_map_chunk_start_index=i,
                                                           pdf=pdf,
                                                           layername=layername,
                                                           im=im,
                                                           vmin=vmin,
                                                           vmax=vmax,
                                                           colormap=colormap,
                                                           figsize=figsize)
                i += len(feature_map_chunk)
                del feature_map_chunk # Cleanup references before closing shared memory buffer

            del feature_map_chunks # Cleanup references before closing shared memory buffer

        del feature_map # Cleanup references before closing shared memory buffer
        shared_buf.close()

        logger.info("%s for layer '%s' finished in %s s.", multiprocessing.current_process().name, layername, time.time() - start)

        return outfile_pdf

    @classmethod
    def _gen_pdf(cls,
                  feature_maps: OrderedDict[str, numpy.typing.NDArray[np.float32]],
                  figsize: tuple[float, float],
                  outdir: Path,
                  basename: str) -> None:
        from pypdf import PdfWriter

        merger = PdfWriter()

        # Need to use spawn instead of fork for matplotlib to actually work in parallel
        with SharedMemoryManager() as smm, ProcessPoolExecutor(mp_context=multiprocessing.get_context('spawn'),
                                                               initializer=cls._init_process) as executor:
            futures: list[Future[Path]] = []
            for layername, feature_map in feature_maps.items():
                # Create shared memory buffer for each layer to avoid having to pickle and send large arrays
                shared_buf = smm.SharedMemory(size=feature_map.nbytes)
                data_shared = np.frombuffer(shared_buf.buf, dtype=feature_map.dtype).reshape(feature_map.shape)
                np.copyto(data_shared, feature_map)
                del data_shared

                futures.append(executor.submit(cls._gen_pdf_layer,
                                           layername,
                                           shared_buf,
                                           feature_map.dtype,
                                           feature_map.shape,
                                           outdir/f'{basename}_{layername}.pdf',
                                           figsize))

            for f in futures:
                merger.append(f.result())

        _ = merger.write(outdir/f'{basename}.pdf')
        merger.close()

    @classmethod
    def _gen_pdf_from_featuremap_file(cls, filename: Path) -> None:
        logger.info("Loading '%s'…", filename)
        with filename.open('rb') as f:
            raw_data = f.read()

        if filename.suffix == '.npy':
            feature_maps = np.load(filename, allow_pickle=True)
        elif filename.suffix == '.zst':
            if 'tensorflow' in sys.modules:
                logger.warning("Loading a compressed '.zst' file when TensorFlow is installed may crash!")
            uncompressed_data = blosc2.decompress2(raw_data)
            feature_maps = pickle.loads(uncompressed_data)  # noqa: S301 We need to load a binary-serialized dict of numpy arrays
        else:
            logger.error("'%s' must have '.npy' (uncompressed) or '.zst' (compressed) extension")
            raise ValueError

        if not isinstance(feature_maps, dict):
            logger.error("Found object of type '%s'", type(feature_maps))
            logger.error("'%s' must contain a 'dict' of 'str' keys and 'numpy.ndarray' values", filename)
            raise TypeError
        feature_maps = cast(dict[Any, Any], feature_maps)

        for layername, feature_map in feature_maps.items():
            if not isinstance(layername, str):
                logger.error("Found key of type '%s'", type(layername))
                logger.error("'%s' must contain a 'dict' of 'str' keys and 'numpy.ndarray' values")
                raise TypeError
            if not isinstance(feature_map, np.ndarray):
                logger.error("Found value of type '%s'", type(feature_map))
                logger.error("'%s' must contain a 'dict' of 'str' keys and 'numpy.ndarray' values")
                raise TypeError

        basename = filename.stem
        outdir = cls.outdir/basename
        outdir.mkdir(parents=True, exist_ok=True)

        cls._gen_pdf(feature_maps, figsize=cls.figsize, outdir=outdir, basename=basename)

    @classmethod
    def gen_featuremap_pdf(cls) -> None:
        from qualia_core.utils.logger.setup_root_logger import setup_root_logger

        # We main not be called from qualia_core.main:main so always setup logging to show logger.info()
        setup_root_logger(colored=True)

        if len(sys.argv) != 2:  # noqa: PLR2004
            logger.error('Invalid arguments.\nUsage: %s <filename>', sys.argv[0])
            sys.exit(1)

        filename = Path(sys.argv[1])
        cls._gen_pdf_from_featuremap_file(filename)

    @override
    def __call__(self,
                 trainresult: TrainResult,
                 model_conf: ModelConfigDict) -> tuple[TrainResult, ModelConfigDict]:
        outdir = self.outdir/trainresult.name
        basename = f'{trainresult.name}_r{trainresult.i}'

        outdir.mkdir(parents=True, exist_ok=True)

        feature_maps = self.__gen_featuremaps(trainresult, self.__data_range)
        if feature_maps is None:
            return trainresult, model_conf

        if self.__save_feature_maps:
            if self.__compress_feature_maps:
                data = pickle.dumps(feature_maps)
                compressed_data = blosc2.compress2(data, codec=blosc2.Codec.ZSTD, clevel=5, nthreads=os.cpu_count())
                with (outdir/f'{basename}.zst').open('wb') as f:
                    _ = f.write(compressed_data)
            else:
                np.save(outdir/f'{basename}.npy', feature_maps, allow_pickle=True)

        if self.__create_pdf:
            self._gen_pdf(feature_maps, figsize=self.figsize, outdir=outdir, basename=basename)

        return trainresult, model_conf
