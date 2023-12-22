from __future__ import annotations

import sys
from typing import Any, cast

import torch
from torch.autograd.function import Function, FunctionCtx

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class IntNoGradientceil(Function):
    @staticmethod
    @override
    def forward(ctx: FunctionCtx, *args: torch.Tensor, **_: Any) -> torch.Tensor:
        return args[0].ceil()

    @staticmethod
    @override
    def backward(ctx: Function, *grad_outputs: torch.Tensor) -> tuple[torch.Tensor | None, None]:
        return grad_outputs[0], None

    @classmethod
    @override
    def apply(cls, *args: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return cast(torch.Tensor, super().apply(*args, **kwargs))  # type: ignore[no-untyped-call]

class IntNoGradientround(Function):
    @staticmethod
    @override
    def forward(ctx: FunctionCtx, *args: torch.Tensor, **_: Any) -> torch.Tensor:
        """Round to nearest.

        Half tie-breaker always rounds upwards to match Qualia-CodeGen's implementation,
        easier to implement than :meth:`torch.round`'s "round half to even" but seems to provide similar results.
        """
        return (args[0] + 0.5).floor()

    @staticmethod
    @override
    def backward(ctx: Function, *grad_outputs: torch.Tensor) -> tuple[torch.Tensor | None, None]:
        return grad_outputs[0], None

    @classmethod
    @override
    def apply(cls, *args: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return cast(torch.Tensor, super().apply(*args, **kwargs))  # type: ignore[no-untyped-call]

class IntNoGradientfloor(Function):
    @staticmethod
    @override
    def forward(ctx: FunctionCtx, *args: torch.Tensor, **_: Any) -> torch.Tensor:
        return args[0].floor()

    @staticmethod
    @override
    def backward(ctx: Function, *grad_outputs: torch.Tensor) -> tuple[torch.Tensor | None, None]:
        return grad_outputs[0], None

    @classmethod
    @override
    def apply(cls, *args: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return cast(torch.Tensor, super().apply(*args, **kwargs))  # type: ignore[no-untyped-call]
