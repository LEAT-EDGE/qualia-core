from __future__ import annotations

import logging
import sys
from math import pi, sin

from torch.optim.lr_scheduler import LRScheduler

from qualia_core.typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.optim.optimizer import Optimizer

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)


class SinDescent(LRScheduler):
    def __init__(self,  # noqa: PLR0913
                 optimizer: Optimizer,
                 epoch: int = -1,
                 pme: int = 0,
                 w: int = 1,
                 lr0: float = 0.1,
                 lrf: float = 0.1,
                 last_epoch: int = -1) -> None:
        self.epoch = epoch
        self.pme = pme
        self.w = w
        self.lr0 = lr0
        self.lrf = lrf
        super().__init__(optimizer, last_epoch)

    @override
    def get_lr(self) -> list[float]:
        if not self._get_lr_called_within_step:
            logger.warning('To get the last learning rate computed by the scheduler, please use `get_last_lr()`.')
        return [self.sin_d(self._step_count - 1) for _ in self.optimizer.param_groups]

    def _get_closed_form_lr(self) -> list[float]:
        return [self.sin_d(base_lr) for base_lr in self.base_lrs]

    def sin_d(self, lr: float) -> float:
        return self.lts(lr) * self.sf(lr) + self.lg_f(lr)

    def lts(self, lr: float) -> float:
        return (-lr / self.epoch + 1) * self.pme

    def sf(self, lr: float) -> float:
        return sin(lr / self.epoch * 2 * self.w * pi)

    def lg_f(self, lr: float) -> float:
        ovs = (-self.epoch * self.lr0) / (self.lrf - self.lr0)
        return self.lr0 * (-lr / ovs + 1)
