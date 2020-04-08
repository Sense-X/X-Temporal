import copy
import logging

import torch

logger = logging.getLogger('global')


def load_state_dict(model, other_state_dict, strict=False):
    """
    1. load resume model or pretained detection model
    2. load pretrained clssification model
    """
    def remove_prefix(state_dict, prefix):
        """Old style model is stored with all names of parameters share common prefix 'module.'"""
        def f(x): return x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        return {f(key): value for key, value in state_dict.items()}
    logger.info(
        "try to load the whole resume model or pretrained detection model...")

    model_state_dict = model.state_dict()
    model_keys = model_state_dict.keys()
    if not list(model_keys)[0].startswith('module'):
        other_state_dict = remove_prefix(other_state_dict, 'module.')
    other_keys = other_state_dict.keys()
    shared_keys, unexpected_keys, missing_keys \
        = check_keys(model_keys, other_keys, 'model')

    incompatible_keys = set([])
    for key in other_keys:
        if key in model_keys:
            if model_state_dict[key].shape != other_state_dict[key].shape:
                incompatible_keys.add(key)

    for key in incompatible_keys:
        other_state_dict.pop(key)
    unexpected_keys = unexpected_keys & incompatible_keys
    model.load_state_dict(other_state_dict, strict=strict)

    num_share_keys = len(shared_keys)
    display_info("model", shared_keys, unexpected_keys, missing_keys)
    if num_share_keys == 0:
        logger.info(
            'failed to load the whole detection model directly,'
            'try to load each part seperately...')
        for mname, module in model.named_children():
            module.load_state_dict(other_state_dict, strict=strict)
            module_keys = module.state_dict().keys()
            other_keys = other_state_dict.keys()

            # check and display info module by module
            shared_keys, unexpected_keys, missing_keys, \
                = check_keys(module_keys, other_keys, mname)
            display_info(mname, shared_keys, unexpected_keys, missing_keys)
            num_share_keys += len(shared_keys)
    else:
        display_info("model", shared_keys, unexpected_keys, missing_keys)
    return num_share_keys


def check_keys(own_keys, other_keys, own_name):
    own_keys = set(own_keys)
    other_keys = set(other_keys)
    shared_keys = own_keys & other_keys
    unexpected_keys = other_keys - own_keys
    missing_keys = own_keys - other_keys
    return shared_keys, unexpected_keys, missing_keys


def display_info(mname, shared_keys, unexpected_keys, missing_keys):
    info = "load {}:{} shared keys, {} unexpected keys, {} missing keys.".format(
        mname, len(shared_keys), len(unexpected_keys), len(missing_keys))

    if len(missing_keys) > 0:
        info += "\nmissing keys are as follows:\n    {}".format(
            "\n    ".join(missing_keys))
    logger.info(info)
