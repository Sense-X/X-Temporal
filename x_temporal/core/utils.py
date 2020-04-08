import numpy as np
import torch
import os


def softmax(scores):
    es = np.exp(scores - scores.max(axis=-1)[..., None])
    return es / es.sum(axis=-1)[..., None]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_bias(net, checkpoint_dir, epoch, iters):
    weight_data = {}
    bias_data = {}
    idx = 0
    for layer_id in range(1, 5):
        layer = getattr(net, 'layer' + str(layer_id))
        blocks = list(layer.children())
        for i, b in enumerate(blocks):
            bias_data[idx] = blocks[i].conv1.buffer[0]
            weight_data[idx] = blocks[i].conv1.buffer[1]
            idx += 1
    w_save_path = os.path.join(
        checkpoint_dir, 'data', '%d_%d_weight.npz' %
        (epoch, iters))
    b_save_path = os.path.join(
        checkpoint_dir, 'data', '%d_%d_bias.npz' %
        (epoch, iters))
    np.savez(w_save_path, weight_data)
    np.savez(b_save_path, bias_data)
