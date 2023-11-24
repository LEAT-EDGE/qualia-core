import torch
from math import sin, pi
import torch.optim.lr_scheduler as sch

class SinDescent(sch._LRScheduler):
    def __init__(self, optimizer, epoch=-1, pme=0, w=1, lr0=0.1, lrf = 0.1, last_epoch=-1, verbose=False):
        self.epoch  = epoch
        self.pme    = pme
        self.w      = w
        self.lr0    = lr0
        self.lrf   = lrf
        super().__init__(optimizer, last_epoch, verbose)
    

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        return [self.sinD(self._step_count-1) for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [self.sinD(base_lr) for base_lr in self.base_lrs]
    
    def sinD(self, lr):
        return self.lts(lr)*self.sf(lr) + self.LGf(lr)
    
    def lts(self, lr):
        return (-lr/self.epoch + 1)* self.pme
    
    def sf(self, lr):
        return sin(lr/self.epoch *2*self.w*pi)
    
    def LGf(self, lr):
        ovs = (-self.epoch*self.lr0)/(self.lrf-self.lr0)
        return self.lr0*(-lr/ovs +1)
