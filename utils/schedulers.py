import torch
import math
from torch.optim.lr_scheduler import _LRScheduler

def get_scheduler(optimizer, args):

    if args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    gamma=0.1,
                                                    step_size=150)

    elif args.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=50)

    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=args.epochs, eta_min=args.lr * 1e-3)

    elif args.scheduler == 'cosine_warm_restarts':

        try: num_restarts = args.num_restarts
        except: print('Warning: Num restarts not specified...using 2'); num_restarts = 2

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                         T_0=int(args.epochs / (num_restarts + 1)),
                                                                         eta_min=args.lr * 1e-3)

    elif args.scheduler == 'cosine_warm_restarts_warmup':

        try: num_restarts = args.num_restarts
        except: print('Warning: Num restarts not specified...using 2'); num_restarts = 2

        scheduler = CosineAnnealingWarmupRestarts_New(warmup_epochs=10, optimizer=optimizer,
                                                                        T_0=int(args.epochs / (num_restarts + 1)),
                                                                         eta_min=args.lr * 1e-3)

    elif args.scheduler == 'warm_restarts_plateau':
        scheduler = WarmRestartPlateau(T_restart=120, optimizer=optimizer, threshold_mode='abs', threshold=0.5,
                                                               mode='min', patience=100)

    elif args.scheduler == 'multi_step':

        try:

            steps = args.steps

        except:

            print('Warning: No step list for Multi-Step Scheduler, using constant step of 30 epochs')
            steps = [30 * i for i in range(1, 5)]

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps)

    else:

        raise NotImplementedError

    return scheduler


class WarmRestartPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):

    """
    Reduce learning rate on plateau and reset every T_restart epochs
    """

    def __init__(self, T_restart, *args, ** kwargs):

        super().__init__(*args, **kwargs)

        self.T_restart = T_restart
        self.base_lrs = [group['lr'] for group in self.optimizer.param_groups]

    def step(self, *args, **kwargs):

        super().step(*args, **kwargs)

        if self.last_epoch > 0 and self.last_epoch % self.T_restart == 0:

            for group, lr in zip(self.optimizer.param_groups, self.base_lrs):
                group['lr'] = lr

            self._reset()


class CosineAnnealingWarmupRestarts_New(torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):

    def __init__(self, warmup_epochs, *args, **kwargs):

        super(CosineAnnealingWarmupRestarts_New, self).__init__(*args, **kwargs)

        # Init optimizer with low learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.eta_min

        self.warmup_epochs = warmup_epochs

        # Get target LR after warmup is complete
        target_lr = self.eta_min + (self.base_lrs[0] - self.eta_min) * (1 + math.cos(math.pi * warmup_epochs / self.T_i)) / 2

        # Linearly interpolate between minimum lr and target_lr
        linear_step = (target_lr - self.eta_min) / self.warmup_epochs
        self.warmup_lrs = [self.eta_min + linear_step * (n + 1) for n in range(warmup_epochs)]

    def step(self, epoch=None):

        # Called on super class init
        if epoch is None:
            super(CosineAnnealingWarmupRestarts_New, self).step(epoch=epoch)

        else:
            if epoch < self.warmup_epochs:
                lr = self.warmup_lrs[epoch]
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr

                # Fulfill misc super() funcs
                self.last_epoch = math.floor(epoch)
                self.T_cur = epoch
                self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

            else:

                super(CosineAnnealingWarmupRestarts_New, self).step(epoch=epoch)