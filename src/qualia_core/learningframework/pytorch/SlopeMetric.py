from __future__ import annotations

import sys
from typing import Any

import torch
from torchmetrics.metric import Metric

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override


class SlopeMetric(Metric):
    count: torch.Tensor
    sum_x: torch.Tensor
    sum_y: torch.Tensor
    sum_xy: torch.Tensor
    sum_squared_x: torch.Tensor
    sum_squared_y: torch.Tensor

    def __init__(self, **kwargs: dict[str, Any]) -> None:
        super().__init__(**kwargs)
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('sum_x', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('sum_y', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('sum_xy', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('sum_squared_x', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('sum_squared_y', default=torch.tensor(0.0), dist_reduce_fx='sum')

    @override
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        self.count += preds.size(0)

        self.sum_x += preds.sum()
        self.sum_y += target.sum()

        self.sum_xy += (preds * target).sum()

        self.sum_squared_x += (preds * preds).sum()
        self.sum_squared_y += (target * target).sum()

    @override
    def compute(self) -> torch.Tensor:
        mean_x = self.sum_x / self.count
        mean_y = self.sum_y / self.count

        squared_mean_y = mean_y * mean_y

        mean_squared_y = self.sum_squared_y / self.count

        var_y = mean_squared_y - squared_mean_y

        mean_xy = self.sum_xy / self.count
        mean_x_mean_y = mean_x * mean_y
        covar_xy = mean_xy - mean_x_mean_y

        return covar_xy / var_y  # slope
