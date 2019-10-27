from torch.optim.lr_scheduler import _LRScheduler


class Scheduler(_LRScheduler):
    def __init__(self, optimizer, mi, mi_I, mi_F, mi_N):
        self.mi = mi
        self.mi_I = mi_I
        self.mi_F = mi_F
        self.mi_N = mi_N
        super(Scheduler, self).__init__(optimizer)

    def get_lr(self):
        return [self.mi(self.mi_I, self.mi_F, self.mi_N, self.last_epoch) for _ in
                self.base_lrs]
