from x_temporal.core.models import TSN
from x_temporal.core.transforms import *
from x_temporal.models.stresnet import *
from x_temporal.models.slowfast import *
from x_temporal.models.resnet3D import *

import torchvision


def get_model(config):
    num_class = config.dataset.num_class
    dropout = config.net.dropout
    arch = config.net.arch

    if config.net.model_type == '2D':
        model = TSN(num_class, config.dataset.num_segments, config.dataset.modality,
                    base_model=config.net.arch,
                    consensus_type=config.net.consensus_type,
                    dropout=config.net.dropout,
                    img_feature_dim=config.net.img_feature_dim,
                    partial_bn=not config.trainer.no_partial_bn,
                    is_shift=config.net.get('shift', False), shift_div=config.net.get('shift_div', 8),
                    non_local=config.net.get('non_local', False),
                    tin=config.net.get('tin', False),
                    pretrain=config.net.get('pretrain', True),
                    )

    elif config.net.model_type == '3D':
        if arch == 'stresnet18':
            model = stresnet18(sample_size=config.dataset.crop_size, sample_duration=config.dataset.num_segments,
                               num_classes=num_class, max_pooling=config.net.max_pooling, dropout=dropout)
        elif arch == 'stresnet50':
            model = stresnet50(sample_size=config.dataset.crop_size, sample_duration=config.dataset.num_segments,
                               num_classes=num_class, max_pooling=config.net.max_pooling, dropout=dropout)
        elif arch == 'stresnet101':
            model = stresnet101(sample_size=config.dataset.crop_size, sample_duration=config.dataset.num_segments,
                                num_classes=num_class, max_pooling=config.net.max_pooling, dropout=dropout)
        elif arch == 'sfresnet50':
            model = sfresnet50(sample_size=config.dataset.crop_size, sample_duration=config.dataset.num_segments,
                                num_classes=num_class, dropout=dropout)
        elif arch == 'sfresnet101':
            model = sfresnet101(sample_size=config.dataset.crop_size, sample_duration=config.dataset.num_segments,
                                num_classes=num_class, dropout=dropout)
        elif arch == 'resnet3D18':
            model = resnet3D18(sample_size=config.dataset.crop_size, sample_duration=config.dataset.num_segments,
                               num_classes=num_class, dropout=dropout)
        elif arch == 'resnet3D50':
            model = resnet3D50(sample_size=config.dataset.crop_size, sample_duration=config.dataset.num_segments,
                               num_classes=num_class, dropout=dropout)
        else:
            raise ValueError("Not Found Arch: %s" % arch)

    cur_device = torch.cuda.current_device()
    model = model.cuda(device=cur_device)
    if config.gpus > 1:
        model = torch.nn.parallel.DistributedDataParallel(
                module=model, device_ids=[cur_device], output_device=cur_device, find_unused_parameters=True
                )

    return model


def get_augmentation(config):
    if config.dataset.modality == 'RGB':
        if config.dataset.flip:
            return torchvision.transforms.Compose([GroupMultiScaleCrop(config.dataset.crop_size, [1, .875, .75, .66]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
        else:
            return torchvision.transforms.Compose(
                [GroupMultiScaleCrop(config.dataset.crop_size, [1, .875, .75, .66])])
    elif config.dataset.modality == 'Flow':
        return torchvision.transforms.Compose([GroupMultiScaleCrop(config.dataset.crop_size, [1, .875, .75]),
                                               GroupRandomHorizontalFlip(is_flow=True)])
    elif config.dataset.modality == 'RGBDiff':
        return torchvision.transforms.Compose([GroupMultiScaleCrop(config.dataset.crop_size, [1, .875, .75]),
                                               GroupRandomHorizontalFlip(is_flow=False)])


def get_optim_policies(model, args):
    first_conv_weight = []
    first_conv_bias = []
    normal_weight = []
    normal_bias = []
    lr5_weight = []
    lr10_bias = []
    cs_weight = []
    cs_bias = []
    bn = []
    custom_ops = []

    conv_cnt = 0
    bn_cnt = 0
    linear_cnt = 0
    fc_lr5 = not (args.tune_from and args.dataset in args.tune_from),
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d) or isinstance(
                m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv3d):
            ps = list(m.parameters())
            conv_cnt += 1
            if conv_cnt == 1:
                first_conv_weight.append(ps[0])
                if len(ps) == 2:
                    first_conv_bias.append(ps[1])
            else:
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
        elif isinstance(m, torch.nn.Linear) and (not args.is_dtn):
            ps = list(m.parameters())
            if fc_lr5:
                lr5_weight.append(ps[0])
            else:
                normal_weight.append(ps[0])
            if len(ps) == 2:
                if fc_lr5:
                    lr10_bias.append(ps[1])
                else:
                    normal_bias.append(ps[1])

        elif isinstance(m, torch.nn.Linear) and args.is_dtn:
            linear_cnt += 1
            ps = list(m.parameters())

            if linear_cnt < 33:
                cs_weight.append(ps[0])
                cs_bias.append(ps[1])
            else:
                if fc_lr5:
                    lr5_weight.append(ps[0])
                else:
                    normal_weight.append(ps[0])
                if len(ps) == 2:
                    if fc_lr5:
                        lr10_bias.append(ps[1])
                    else:
                        normal_bias.append(ps[1])

        elif isinstance(m, torch.nn.BatchNorm2d):
            bn_cnt += 1
            # later BN's are frozen
            if args.no_partialbn or bn_cnt == 1:
                bn.extend(list(m.parameters()))
        elif isinstance(m, torch.nn.BatchNorm3d):
            bn_cnt += 1
            # later BN's are frozen
            if args.no_partialbn or bn_cnt == 1:
                bn.extend(list(m.parameters()))
        elif len(m._modules) == 0:
            if len(list(m.parameters())) > 0:
                raise ValueError(
                    "New atomic module type: {}. Need to give it a learning policy".format(
                        type(m)))

    return [
        {'params': first_conv_weight, 'lr_mult': 5 if args.modality == 'Flow' else 1, 'decay_mult': 1,
            'name': "first_conv_weight"},
        {'params': first_conv_bias, 'lr_mult': 10 if args.modality == 'Flow' else 2, 'decay_mult': 0,
            'name': "first_conv_bias"},
        {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
            'name': "normal_weight"},
        {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
            'name': "normal_bias"},
        {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
            'name': "BN scale/shift"},
        {'params': custom_ops, 'lr_mult': 1, 'decay_mult': 1,
            'name': "custom_ops"},
        # for fc
        {'params': lr5_weight, 'lr_mult': 5, 'decay_mult': 1,
            'name': "lr5_weight"},
        {'params': lr10_bias, 'lr_mult': 10, 'decay_mult': 0,
            'name': "lr10_bias"},
        {'params': cs_weight, 'lr_mult': 1, 'decay_mult': 1,
            'name': "cs_weight"},
        {'params': cs_bias, 'lr_mult': 2, 'decay_mult': 0,
            'name': "cs_bias"},
    ]
