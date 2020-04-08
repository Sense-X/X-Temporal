import torch
import copy


def build_cls_instance(module, cfg):
    """Build instance for given cls"""
    cls = getattr(module, cfg['type'])
    return cls(**cfg['kwargs'])


def build_optimizer(cfg_optim, model):
    cfg_optim = copy.deepcopy(cfg_optim)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    cfg_optim['kwargs']['params'] = trainable_params
    optim_type = cfg_optim['type']
    optimizer = build_cls_instance(torch.optim, cfg_optim)
    return optimizer
