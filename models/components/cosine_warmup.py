from torch.optim.lr_scheduler import _LRScheduler
import math

class CosineWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1, min_lr=1e-6):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr

        # Store initial lr, min_lr, and lr_scale for each param group
        self.init_lrs = []
        self.min_lrs = []
        self.lr_scales = []
        for param_group in optimizer.param_groups:
            self.init_lrs.append(param_group.get('initial_lr', param_group['lr'])) #could be kept for later use when doing per group lrs
            self.min_lrs.append(param_group.get('min_lr', self.min_lr)) # could be kept for later use when doing per group lrs
            self.lr_scales.append(param_group.get('lr_scale', 1.0))
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(0, self.last_epoch)
        lrs = []
        for idx, (init_lr, min_lr, lr_scale) in enumerate(zip(self.init_lrs, self.min_lrs, self.lr_scales)):
            if step < self.warmup_steps:
                lr = init_lr * step / float(max(1, self.warmup_steps))
            else:
                progress = float(step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
                lr = min_lr + (init_lr - min_lr) * 0.5 * (1. + math.cos(math.pi * progress))
            lr *= lr_scale
            lrs.append(lr)
        return lrs