from torch.optim.lr_scheduler import _LRScheduler

from Utils import mi


class Scheduler(_LRScheduler):
    def __init__(self, optimizer):
        super(Scheduler, self).__init__(optimizer)

    def get_lr(self):
        return [mi(self.last_epoch) for _ in self.base_lrs]