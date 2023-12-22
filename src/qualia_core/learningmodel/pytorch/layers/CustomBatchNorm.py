import sys
from abc import ABC
from typing import Any, Optional

import torch
from torch import nn

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class CustomBatchNorm(ABC, nn.Module):
    eps: float
    momentum: float
    affine: bool
    track_running_stats: bool
    weight: nn.Parameter
    bias: nn.Parameter
    running_var: nn.Parameter
    running_mean: nn.Parameter
    num_batches_tracked: nn.Parameter
    use_fused_params: bool

    def __init__(self,
                 num_features: int,
                 eps: float = 1e-05,
                 momentum: float = 0.1,
                 affine: bool = True,
                 track_running_stats: bool = True,
                 use_fused_params: bool = True,
                 device: Any = None,
                 dtype: Any = None) -> None:
        self.call_super_init = True # Support multiple inheritance from nn.Module
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.use_fused_params = use_fused_params

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features, device=device, dtype=dtype))
            self.bias = nn.Parameter(torch.zeros(num_features, device=device, dtype=dtype))

        if track_running_stats:
            self.running_var = nn.Parameter(torch.ones(num_features, device=device, dtype=dtype), requires_grad=False)
            self.running_mean = nn.Parameter(torch.zeros(num_features, device=device, dtype=dtype), requires_grad=False)
            self.num_batches_tracked = nn.Parameter(torch.tensor(0, device=device, dtype=torch.long), requires_grad=False)

    @override
    def extra_repr(self) -> str:
        return (f'{self.num_features}, eps={self.eps}, momentum={self.momentum}, affine={self.affine}, '
                f'track_running_stats={self.track_running_stats}')

    def update_running_stats(self, mean: torch.Tensor, var: torch.Tensor) -> None:
        """Update Exponential Moving Average for input mean and variance."""
        with torch.no_grad():
            self.num_batches_tracked += 1
            self.running_mean *= (1 - self.momentum)
            self.running_mean += self.momentum * mean
            self.running_var *= (1 - self.momentum)
            self.running_var += self.momentum * var

    def compute_stats(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute mean and variance for each channel of input."""
        mean = x.mean(dim=(0, -1))
        var = x.var(dim=(0, -1), correction=0) # PyTorch BatchNorm1d doc unbiased=False, correction=0 is PyTorch 2.0 equivalent
        return mean, var

    def compute_fused_params(self,
                             weight: Optional[torch.Tensor],
                             bias: Optional[torch.Tensor],
                             mean: torch.Tensor,
                             var: torch.Tensor,
                             eps: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert the 4(+1) params (weight, bias, mean, variance, +eps) to 2 params to be applied with y = alpha × x + beta.""" # noqa: RUF002
        sigma = (var + eps).sqrt()

        alpha = (weight if weight is not None else 1) / sigma
        beta = (bias if bias is not None else 0) - (weight if weight is not None else 1) * mean / sigma

        return alpha, beta

    def compute_batchnorm_with_fused_params(self,
                                       x: torch.Tensor,
                                       alpha: torch.Tensor,
                                       beta: torch.Tensor) -> torch.Tensor:
        """Compute y = alpha × x + beta.""" # noqa: RUF002
        return x * alpha.unsqueeze(-1) + beta.unsqueeze(-1)

    def compute_batchnorm(self,
                          x: torch.Tensor,
                          weight: Optional[torch.Tensor],
                          bias: Optional[torch.Tensor],
                          mean: torch.Tensor,
                          var: torch.Tensor,
                          eps: float) -> torch.Tensor:
        """Compute batchnorm for input x with given weight, bias, mean, variance and epsilon."""
        if self.use_fused_params:
            alpha, beta = self.compute_fused_params(weight=weight, bias=bias, mean=mean, var=var, eps=eps)
            return self.compute_batchnorm_with_fused_params(x=x, alpha=alpha, beta=beta)

        y = (x - mean.unsqueeze(-1)) / (var.unsqueeze(-1) + eps).sqrt()
        if weight is not None:
            y = y * weight.unsqueeze(-1)
        if bias is not None:
            y = y + bias.unsqueeze(-1)
        return y

    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """BatchNorm1d in Python PyTorch.

        Update statistics if required then compute batchnorm for given input using module's own weights, bias and running stats.
        """
        mean, var = self.compute_stats(x=input)

        if self.training and self.track_running_stats:
            self.update_running_stats(mean=mean, var=var)

        if self.training or not self.track_running_stats:
            y = self.compute_batchnorm(x=input,
                                       weight=self.weight if self.affine else None,
                                       bias=self.bias if self.affine else None,
                                       mean=mean,
                                       var=var,
                                       eps=self.eps)
        else:
            y = self.compute_batchnorm(x=input,
                                       weight=self.weight if self.affine else None,
                                       bias=self.bias if self.affine else None,
                                       mean=self.running_mean,
                                       var=self.running_var,
                                       eps=self.eps)
        return y
