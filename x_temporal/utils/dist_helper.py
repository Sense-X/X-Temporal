import os
import logging
import shutil
import torch
import math
import numpy as np
from collections import defaultdict
import torch.distributed as dist



def load_state(path, model, optimizer=None):

    rank = get_rank()

    def map_func(storage, location):
        return storage.cuda()

    if os.path.isfile(path):
        if rank == 0:
            print("=> loading checkpoint '{}'".format(path))

        checkpoint = torch.load(path, map_location=map_func)
        model.load_state_dict(checkpoint['state_dict'], strict=False)

        if rank == 0:
            ckpt_keys = set(checkpoint['state_dict'].keys())
            own_keys = set(model.state_dict().keys())
            missing_keys = own_keys - ckpt_keys
            for k in missing_keys:
                print(
                    'caution: missing keys from checkpoint {}: {}'.format(
                        path, k))

        if optimizer is not None:
            best_prec1 = checkpoint['best_prec1']
            last_iter = checkpoint['step']
            optimizer.load_state_dict(checkpoint['optimizer'])
            if rank == 0:
                print(
                    "=> also loaded optimizer from checkpoint '{}' (iter {})".format(
                        path, last_iter))
            return best_prec1, last_iter
    else:
        if rank == 0:
            print("=> no checkpoint found at '{}'".format(path))


def is_master_proc(num_gpus=8):
    """
    Determines if the current process is the master process.
    """
    if torch.distributed.is_initialized():
        return dist.get_rank() % num_gpus == 0
    else:
        return True


def all_gather(tensors):
    """
    All gathers the provided tensors from all processes across machines.
    Args:
        tensors (list): tensors to perform all gather across all processes in
        all machines.
    """

    gather_list = []
    output_tensor = []
    world_size = dist.get_world_size()
    for tensor in tensors:
        tensor_placeholder = [
            torch.ones_like(tensor) for _ in range(world_size)
        ]
        dist.all_gather(tensor_placeholder, tensor, async_op=False)
        gather_list.append(tensor_placeholder)
    for gathered_tensor in gather_list:
        output_tensor.append(torch.cat(gathered_tensor, dim=0))
    return output_tensor


def all_reduce(tensor, average=True):
    """
    All reduce the provided tensors from all processes across machines.
    Args:
        tensor (tensor): tensor to perform all reduce across all processes in
        all machines.
        average (bool): scales the reduced tensor by the number of overall
        processes across all machines.
    """

    dist.all_reduce(tensor, async_op=False)
    if average:
        world_size = dist.get_world_size()
        tensor.mul_(1.0 / world_size)
    return tensor


def get_world_size():
    """
    Get the size of the world.
    """
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """
    Get the rank of the current process.
    """
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


