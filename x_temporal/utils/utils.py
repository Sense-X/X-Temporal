import json
import os
import re
import copy

import torch
import numpy as np


def format_cfg(cfg):
    """Format experiment config for friendly display"""

    def list2str(cfg):
        for key, value in cfg.items():
            if isinstance(value, dict):
                cfg[key] = list2str(value)
            elif isinstance(value, list):
                if len(value) == 0 or isinstance(value[0], (int, float)):
                    cfg[key] = str(value)
                else:
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            value[i] = list2str(item)
                    cfg[key] = value
        return cfg

    cfg = list2str(copy.deepcopy(cfg))
    json_str = json.dumps(cfg, indent=2, ensure_ascii=False).split(r"\n")
    json_str = [re.sub(r"(\"|,$|\{|\}|\[$)", "", line)
                for line in json_str if line.strip() not in "{}[]"]
    cfg_str = r"\n".join([line.rstrip() for line in json_str if line.strip()])
    return cfg_str


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.contiguous().view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            #assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count


def load_checkpoint(ckpt_path):
    """Load state_dict from checkpoint"""

    def remove_prefix(state_dict, prefix):
        """Old style model is stored with all names of parameters share common prefix 'module.'"""
        def f(x): return x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        return {f(key): value for key, value in state_dict.items()}

    assert os.path.exists(ckpt_path), f'No such file: {ckpt_path}'
    device = torch.cuda.current_device()
    ckpt_dict = torch.load(
        ckpt_path,
        map_location=lambda storage,
        loc: storage.cuda(device))

    # handle different storage format between pretrain vs resume
    if 'model' in ckpt_dict:
        state_dict = ckpt_dict['model']
    elif 'state_dict' in ckpt_dict:
        state_dict = ckpt_dict['state_dict']
    else:
        state_dict = ckpt_dict

    state_dict = remove_prefix(state_dict, 'module.')
    ckpt_dict['model'] = state_dict

    return ckpt_dict
