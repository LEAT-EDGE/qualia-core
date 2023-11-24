# TO DO : Delete

from typing import Optional

import torch


class IntNoGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.floor()

    @staticmethod
    def backward(ctx, g):
        return g

def maxWeight(weight: torch.Tensor, maxif: float, training: bool) -> torch.Tensor:
    maxi = torch.tensor(maxif, device=weight.device)

#    if(maxi==0 or abs(maxi) == float('inf') or training):
#        maxi = torch.max(torch.abs(weight.view(-1)))

    if maxi == 0:
        maxi = torch.tensor(-1, device=weight.device)
    elif abs(maxi) != float('inf'):
        maxi = -(torch.log2(maxi).floor().int() + 1)
    else:
        maxi = torch.tensor(0, device=weight.device)
    return maxi


def quantifier(w: torch.Tensor, n_bit: int, training: bool, maxi: float = 0, force_q: Optional[int] = None) -> torch.Tensor:
    tn_bit = torch.tensor(n_bit, device=w.device)

    if force_q is not None:
        v = torch.pow(2.0, torch.tensor(force_q, device=w.device))
    else:
        qint = maxWeight(w, maxi, training)
        v = torch.pow(2.0, tn_bit - 1 + qint)

    w = w*v
    w = IntNoGradient.apply(w)
    w = torch.clamp(w, -torch.pow(2, tn_bit - 1), torch.pow(2, tn_bit - 1) - 1)
    w = w / v

    return w
