from torch.optim.lr_scheduler import _LRScheduler
from GQN.Utils import mi


class Scheduler(_LRScheduler):
    def __init__(self, optimizer, properties):
        self.properties = properties
        super(Scheduler, self).__init__(optimizer)

    def get_lr(self):
        return [mi(self.properties.mi_I, self.properties.mi_F, self.properties.mi_N, self.last_epoch) for _ in
                self.base_lrs]
