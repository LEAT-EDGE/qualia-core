from __future__ import annotations

import logging
import sys
from typing import Literal, cast, overload

import torch
from torch import nn

from qualia_core.typing import QuantizationConfigDict, QuantizerConfigDict
from qualia_core.typing import QuantizationConfigDict as QuantizationConfig
from qualia_core.typing import QuantizerConfigDict as QuantizerConfig

from .grad_functions import IntNoGradientceil, IntNoGradientfloor, IntNoGradientround
from .range_setting import MSE_analysis_range_setting, MSE_simulation_range_setting, minmax_range_setting

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)

def grad_scale(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


def round_pass(x: torch.Tensor) -> torch.Tensor:
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad

def maxWeight(maxi: torch.Tensor) -> int:
        maxi = torch.tensor(maxi, device=maxi.device)
        if maxi == 0:
            maxi = torch.tensor(-1, device=maxi.device)
        elif abs(maxi) != float('inf'):
            maxi = -(torch.log2(maxi).floor().int() + 1)
        else:
            maxi = torch.tensor(0, device=maxi.device)
        return cast(int, maxi.item()) # maxi is necessarily an int in all 3 cases

def flat_and_cat(w: torch.Tensor, bias_tensor: torch.Tensor | None) -> torch.Tensor:
    w_and_biases = torch.flatten(w)
    if bias_tensor is not None:
        w_and_biases = torch.cat((w_and_biases, torch.flatten(bias_tensor)))
    return w_and_biases

class Quantizer(nn.Module):
    global_min: torch.Tensor
    global_max: torch.Tensor

    def __init__(self,  # noqa: PLR0913
                 quant_enable: bool = True,  # noqa: FBT001, FBT002
                 LSQ: bool = False,  # noqa: FBT001, FBT002
                 bits: int = 8,
                 force_q: int | None = None,
                 range_setting: str = 'minmax',
                 roundtype: str = 'nearest',
                 tensor_type: str | None = None,
                 quantype: str | None = None,
                 is_asymmetric: bool = False) -> None:  # noqa: FBT001, FBT002
        super().__init__()

        self.bits = bits
        self.quantype = quantype
        self.range_setting = range_setting
        self.is_asymmetric = is_asymmetric
        self.LSQ = LSQ
        self.force_q = force_q
        self.roundtype = roundtype
        self.tensor_type = tensor_type
        self.quant_enable = quant_enable
        self.global_max =  torch.tensor(-float('inf'))
        self.global_min =  torch.tensor(+float('inf'))

        # unsigned activation might be used later
        #   self.neg_bits = 0
        #   self.pos_bits = 2 ** self.bits - 1

        # Set LSQscale weight to None, it will be created later
        self.LSQscale = None

        if self.is_asymmetric:
            # asymmetricsigned weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
            self.neg_bits = - 2 ** (self.bits - 1)
            self.pos_bits = 2 ** (self.bits - 1) - 1
        else:
            # symmetric signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
            self.neg_bits = - 2 ** (self.bits - 1)
            self.pos_bits = 2 ** (self.bits - 1) - 1

    @override
    def extra_repr(self) -> str:
        """Add custom arguments to the ``__repr__`` method.

        :return: String representation of :class:`Quantizer` with custom arguments.
        """
        s = super().extra_repr() + ', ' if super().extra_repr() else ''
        s += f'quant_enable={self.quant_enable}'
        s += f', LSQ={self.LSQ}'
        s += f', bits={self.bits}'
        s += f', range_setting={self.range_setting}'
        s += f', roundtype={self.roundtype}'
        s += f', tensor_type={self.tensor_type}'
        s += f', quantype={self.quantype}'
        s += f', is_asymmetric={self.is_asymmetric}'
        if self.force_q is not None:
            s += f', force_q={self.force_q}'
        return s

    @override
    def forward(self,
                w: torch.Tensor,
                bias_tensor: torch.Tensor | None = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # Override the forward method from the base nn.Module class.

        # If quantization is off then return input w +/- bias_tensor
        if not self.quant_enable:
            return w if bias_tensor is None else (w, bias_tensor)

        # Compute/update self.global_max and self.global_min
        self.update_min_and_max(w, bias_tensor)



        ###########################
        ########### LSQ ###########
        if self.LSQ:
            ############### LSQ ###############
            w, bias_tensor, scale, offset = self.applyLSQ(w, bias_tensor)
            ###################################

        else:
        ########### PTQ and QAT ###########
            scale, offset = self.compute_scale_and_offset(flat_and_cat(w, bias_tensor), self.global_max, self.global_min)

            if bias_tensor is None:
                w = self.quant_dequant( w, scale, offset)
            else:
                w, bias_tensor = self.quant_dequant( w, scale, offset), self.quant_dequant( bias_tensor, scale, offset)

        ###################################

        # Return quantized w +/- bias_tensor
        return w if bias_tensor is None else (w, bias_tensor)

    @overload
    def __call__(self,
                 w: torch.Tensor,
                 bias_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    @overload
    def __call__(self,
                 w: torch.Tensor) -> torch.Tensor:
        ...

    @override
    def __call__(self,
                 w: torch.Tensor,
                 bias_tensor: torch.Tensor | None = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return super().__call__(w, bias_tensor=bias_tensor)


    def applyLSQ(self,
                 w: torch.Tensor,
                 bias_tensor: torch.Tensor | None = None)-> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        if self.LSQscale is None:
            w_and_biases = flat_and_cat(w, bias_tensor)
            self.LSQscale = (torch.nn.Parameter(w_and_biases.detach().abs().mean() * 2 / (self.pos_bits ** 0.5))
                             or torch.nn.Parameter(torch.ones(1, device=w.device)))

        if not self.is_asymmetric:
            if self.training:
                # Original here https://arxiv.org/pdf/1902.08153.pdf 2.2 STEP SIZE GRADIENT SCALE
                # s_grad_scale = 1.0 / ((torch.tensor(self.pos_bits, device=w.device) * w.size(1)) ** 0.5) if ("act" in self.tensor_type) or ("input" in self.tensor_type) else 1.0 / ((torch.tensor(self.pos_bits, device=w.device) * w.numel()) ** 0.5)
                # But choosen from  here https://github.com/zhutmost/lsq-net/blob/master/quan/quantizer/lsq.py
                s_grad_scale = 1.0 / ((torch.tensor(self.pos_bits, device=w.device) * w.numel()) ** 0.5)
                scale = grad_scale(self.LSQscale, s_grad_scale)
            else:
                scale = self.LSQscale

            # If scale at 0 put 1
            scale = scale or torch.tensor(1, device=w.device)
            offset = torch.tensor(0, device=w.device)

            # Apply scale and clamp
            w = torch.clamp(w / scale, torch.tensor(self.neg_bits, device=w.device), torch.tensor(self.pos_bits, device=w.device))
            # Apply round and dequantize
            w = round_pass(w) * scale
            if bias_tensor is not None:
                bias_tensor = torch.clamp(bias_tensor / scale,
                                          torch.tensor(self.neg_bits, device=w.device),
                                          torch.tensor(self.pos_bits, device=w.device))
                bias_tensor = round_pass(bias_tensor) * scale
        else:
            raise ValueError('Asym LSQ+ not implemented.')
        return w, bias_tensor, scale, offset

    def update_min_and_max(self, w: torch.Tensor, bias_tensor: torch.Tensor | None = None) -> None:
        # Update only at training
        if self.training:
                # Concatenate weights and biases to be able to compute max and min
                flat_w = flat_and_cat(w, bias_tensor)
                if self.range_setting == 'minmax':
                    max_val, min_val = minmax_range_setting(flat_w, self.is_asymmetric)
                elif self.range_setting == 'MSE_simulation':
                    max_val, min_val = MSE_simulation_range_setting(flat_w, self.bits, self.is_asymmetric)
                elif self.range_setting == 'MSE_analysis':
                    max_val, min_val = MSE_analysis_range_setting(flat_w, self.bits, self.is_asymmetric)
                else:
                    raise ValueError(self.range_setting, 'not implemented.')
                self.global_max = max_val if self.tensor_type in ['w', 'bias'] else torch.max(self.global_max, max_val).detach()
                self.global_min = min_val if self.tensor_type in ['w', 'bias'] else torch.min(self.global_min, min_val).detach()

    def compute_scale_and_offset(self,
                                 w: torch.Tensor,
                                 max_val: torch.Tensor,
                                 min_val: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        max_val = torch.tensor(max_val, device=w.device)
        min_val = torch.tensor(min_val, device=w.device)
        scale = torch.tensor(1, device=w.device)
        offset = torch.tensor(0, device=w.device)
        if max_val != 0 :
            if self.force_q is not None:
                    scale = 1/torch.pow(2.0, torch.tensor(self.force_q, device=w.device))
            elif self.quantype == 'fxp' and not self.is_asymmetric:
                    left_bw = -(torch.log2(max_val).floor().int() + 1)
                    scale = 1/torch.pow(2.0, self.bits - 1 + left_bw)
            elif self.quantype == 'fake':
                max_range = max_val - min_val
                bits_range = float(self.pos_bits - self.neg_bits)
                scale = (max_range) / (bits_range)
            else:
                raise ValueError(self.quantype, 'not implemented. - Does not work with asymmetric.')

        scale = torch.tensor(1, device=w.device) if scale == torch.tensor(0, device=w.device) else scale
        offset = (torch.clamp((-torch.round(min_val / scale) + self.neg_bits), min_val, max_val)
                  if self.is_asymmetric
                  else torch.tensor(0, device=w.device))

        return scale, offset

    def quant_dequant(self, x_f: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        x_int = x_f / scale

        if self.roundtype == 'nearest':
            x_int = IntNoGradientround.apply(x_int)
        elif self.roundtype == 'floor':
            x_int = IntNoGradientfloor.apply(x_int)
        elif self.roundtype == 'ceil':
            x_int = IntNoGradientceil.apply(x_int)
        else:
            raise ValueError(self.roundtype, 'not implemented.')

        if self.is_asymmetric:
            x_int += offset

        x_q = torch.clamp(x_int, torch.tensor(self.neg_bits, device=x_f.device), torch.tensor(self.pos_bits, device=x_f.device))

        if self.is_asymmetric:
            x_q -= offset

        return x_q * scale

    @property
    def fractional_bits(self) -> int | None:
        if not self.quant_enable:
            logger.info('Quantizer for tensor type %s disabled, no fractional bits to return', self.tensor_type)
            return None

        if self.quantype != 'fxp':
            logger.warning("Quantizer type '%s' is not fixed-point, no fractional bits to return", self.quantype)
            return None

        if self.is_asymmetric:
            logger.warning('Asymmetric quantization unsupported, no fractional bits to return')
            return None

        if self.force_q is not None:
            return self.force_q

        return self.bits - 1 + maxWeight(self.global_max)


def update_params(tensor_type: Literal['act', 'v', 'input', 'w', 'bias'],
                  quant_params: QuantizationConfigDict) -> QuantizerConfigDict:
    # Create a copy of quant_params
    quant_params_copy = quant_params.copy()

    # Remove all tensor_type keys from tensor_params
    _ = quant_params_copy.pop('act', None)
    _ = quant_params_copy.pop('v', None)
    _ = quant_params_copy.pop('input', None)
    _ = quant_params_copy.pop('w', None)
    _ = quant_params_copy.pop('bias', None)

    tensor_params = QuantizerConfigDict(quant_params_copy)

    new_params = quant_params.get(tensor_type, None)

    # If tensor_type exists as a key in tensor_params, update tensor_params with the dict associated to the tensor_type key
    if new_params is not None:
        tensor_params.update(new_params)

    # Modify the "tensor_type" key by appending "_type" to keep tensor_type information
    tensor_params['tensor_type'] = tensor_type + '_type'

    # Return the updated tensor_params dictionary
    return tensor_params

__all__ = ['QuantizerConfig', 'QuantizationConfig']
