import logging
import copy

import torch

logger = logging.getLogger('global')


class ExponentialWarmUpLR(object):
    """Scheduler that update learning rate exponentially
    """

    def __init__(self, warmup_iter, init_lr, target_lr):
        self.lr_scale = target_lr / init_lr
        self.gamma = self.lr_scale**(1.0 / max(1, warmup_iter))
        self.warmup_iter = warmup_iter

    def get_lr(self, last_epoch, base_lrs, optimizer):
        return [base_lr * self.gamma**last_epoch /
                self.lr_scale for base_lr in base_lrs]


class LinearWarmUpLR(object):
    """Scheduler that update learning rate linearly
    """

    def __init__(self, warmup_iter, init_lr, target_lr):
        self.lr_gap = target_lr - init_lr
        self.gamma = self.lr_gap / max(1, warmup_iter)
        self.warmup_iter = warmup_iter

    def get_lr(self, last_epoch, base_lrs, optimizer):
        return [base_lr + self.gamma * last_epoch -
                self.lr_gap for base_lr in base_lrs]


_warmup_lr = {
    'linear': LinearWarmUpLR,
    'exp': ExponentialWarmUpLR
}


def build_warmup_scheduler(cfg_scheduler, optimizer, data_size, lr_scale):

    target_lr = [group.get('initial_lr', group['lr'])
                 for group in optimizer.param_groups][0]
    warmup_epochs = cfg_scheduler.get('warmup_epochs', 0)
    # no linear scaling if no warmup
    if warmup_epochs > 0:
        init_lr = target_lr / float(lr_scale)
    else:
        init_lr = target_lr
    warmup_iter = int(warmup_epochs * data_size)

    warmup_type = cfg_scheduler.get('warmup_type', 'exp')
    assert warmup_type in _warmup_lr, f'warmup scheduler {warmup_type} not supported'

    return _warmup_lr[warmup_type](warmup_iter, init_lr, target_lr)


def prepare_scheduler(cfg_scheduler, optimizer, data_size):
    """Convert epoch to iteration"""

    cfg = copy.deepcopy(cfg_scheduler)

    cfg['kwargs']['optimizer'] = optimizer
    if cfg['type'] == 'MultiStepLR':
        cfg['kwargs']['milestones'] = [
            int(e * data_size) for e in cfg['kwargs']['milestones']]
    elif cfg['type'] == 'StepLR':
        cfg['kwargs']['step_size'] = cfg['kwargs']['step_size'] * data_size
    elif cfg['type'] == 'ReduceLROnPlateau':
        cfg['kwargs']['patience'] = cfg['kwargs']['patience'] * data_size
    elif cfg['type'] == 'CosineAnnealingLR':
        cfg['kwargs']['T_max'] = cfg['kwargs']['T_max'] * data_size
    else:
        raise NotImplementedError(f'{cfg} is not supported')
    scheduler = getattr(torch.optim.lr_scheduler, cfg['type'])
    return scheduler, cfg


def build_scheduler(cfg_scheduler, optimizer, data_size, lr_scale):
    """ Build composed warmup scheduler and standard scheduler.
        There will be no linar scaling process if no warmup
    """
    standard_scheduler_class, cfg_scheduler = prepare_scheduler(
        cfg_scheduler, optimizer, data_size)

    warmup_scheduler = build_warmup_scheduler(
        cfg_scheduler, optimizer, data_size, lr_scale)

    class ChainIterLR(standard_scheduler_class):
        """Unified scheduler that chains warmup scheduler and standard scheduler
        """

        def __init__(self, *args, **kwargs):
            super(ChainIterLR, self).__init__(*args, **kwargs)

        def get_lr(self):
            if self.last_iter <= warmup_scheduler.warmup_iter:
                return warmup_scheduler.get_lr(
                    self.last_iter, self.base_lrs, self.optimizer)
            else:
                return super(ChainIterLR, self).get_lr()

        @property
        def last_iter(self):
            return self.last_epoch

    return ChainIterLR(**cfg_scheduler['kwargs'])


if __name__ == '__main__':
    import torchvision
    import os
    import sys
    model = torchvision.models.resnet18()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.1)
    cfg_scheduler = {
        'warmup_epochs': 1,
        'type': 'MultiStepLR',
        'kwargs': {
            'milestones': [10, 20],
            'gamma': 0.1
        }
    }
    scheduler = build_scheduler(
        cfg_scheduler,
        optimizer,
        data_size=5,
        lr_scale=10)
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        state = torch.load(sys.argv[1])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
    start_iter = scheduler.last_iter
    for i in range(start_iter, 120):
        if i % 30 == 0:
            state = {
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()}
            torch.save(state, f'iter{i}.pkl')
        scheduler.step()
        print(f'iter:{i}, lr:{scheduler.get_lr()}')
