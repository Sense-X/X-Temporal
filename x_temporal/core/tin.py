import math
import sys
import random
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from x_temporal.cuda_shift.rtc_wrap import ShiftFeatureFunc

logger = logging.getLogger('global')

def solve_sigmoid(x):
    return -math.log(1.0 / x - 1)


def linear_sampler(data, bias):
    '''
    data: N * T * C * H * W
    bias: N * T * Groups
    weight: N * T
    '''
    N, T, C, H, W = data.shape
    bias_0 = torch.floor(bias).int()
    bias_1 = bias_0 + 1

    # N * T * C * H * W
    sf1 = ShiftFeatureFunc()
    sf2 = ShiftFeatureFunc()

    data = data.view(N, T, C, H * W).contiguous()
    data_0 = sf1(data, bias_0)
    data_1 = sf2(data, bias_1)

    w_0 = 1 - (bias - bias_0.float())
    w_1 = 1 - w_0

    groupsize = bias.shape[1]
    w_0 = w_0[:, :, None].repeat(1, 1, C // groupsize)
    w_0 = w_0.view(w_0.size(0), -1)
    w_1 = w_1[:, :, None].repeat(1, 1, C // groupsize)
    w_1 = w_1.view(w_1.size(0), -1)

    w_0 = w_0[:, None, :, None]
    w_1 = w_1[:, None, :, None]

    out = w_0 * data_0 + w_1 * data_1
    out = out.view(N, T, C, H, W)

    return out


class WeightConvNet(nn.Module):
    def __init__(self, in_channels, groups, n_segment):
        super(WeightConvNet, self).__init__()
        self.lastlayer = nn.Conv1d(in_channels, groups, 3, padding=1)
        self.groups = groups

    def forward(self, x):
        N, C, T = x.shape
        x = self.lastlayer(x)
        x = x.view(N, self.groups, T)
        x = x.permute(0, 2, 1)
        return x


class BiasConvFc2Net(nn.Module):
    def __init__(self, in_channels, groups,
                 n_segment, kernel_size=3, padding=1):
        super(BiasConvFc2Net, self).__init__()
        self.conv = nn.Conv1d(in_channels, 1, kernel_size, padding=padding)
        self.fc = nn.Linear(n_segment, n_segment)
        self.relu = nn.ReLU()
        self.lastlayer = nn.Linear(n_segment, groups)

    def forward(self, x):
        N, C, T = x.shape
        x = self.conv(x)
        x = x.view(N, T)
        x = self.relu(self.fc(x))
        x = self.lastlayer(x)
        x = x.view(N, 1, -1)
        return x


class BiasNet(nn.Module):
    def __init__(self, in_channels, groups, n_segment):
        super(BiasNet, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.net = BiasConvFc2Net(in_channels, groups, n_segment, 3, 1)
        self.net.lastlayer.bias.data[...] = 0.5108

    def forward(self, x):
        x = self.net(x)
        x = 4 * (self.sigmoid(x) - 0.5)
        return x


class WeightNet(nn.Module):
    def __init__(self, in_channels, groups, n_segment):
        super(WeightNet, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.groups = groups * 2

        self.net = WeightConvNet(in_channels, groups, n_segment)

        self.net.lastlayer.bias.data[...] = 0

    def forward(self, x):
        x = self.net(x)
        x = 2 * self.sigmoid(x)
        return x


class TemporalDeform(nn.Module):
    def __init__(self, in_channels, n_segment=3, shift_div=1):
        super(TemporalDeform, self).__init__()
        self.n_segment = n_segment
        self.shift_div = shift_div
        self.in_channels = in_channels

        self.biasnet = BiasNet(in_channels // shift_div, 2, n_segment)
        self.weightnet = WeightNet(in_channels // shift_div, 2, n_segment)

    @staticmethod
    def _init_weights(weights, std):
        fan_out = weights.size(0)
        fan_in = weights.size(1) * weights.size(2)
        w = np.random.normal(0.0, std, (fan_out, fan_in))
        return torch.from_numpy(w.reshape(weights.size()))

    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        fold = c // self.shift_div

        out = torch.zeros_like(x)
        x_def = x[:, :fold, :]

        x_def = x_def.view(n_batch, self.n_segment, fold, h, w)

        x_pooled = torch.mean(x_def, 3)
        x_pooled_1d = torch.mean(x_pooled, 3)
        x_pooled_1d = x_pooled_1d.permute(0, 2, 1).contiguous()
        # N * T * C

        x_bias = self.biasnet(x_pooled_1d).view(n_batch, -1)
        x_weight = self.weightnet(x_pooled_1d)

        x_bias = torch.cat([x_bias, -x_bias], 1)
        x_sa = linear_sampler(x_def, x_bias)

        x_weight = x_weight[:, :, :, None]
        x_weight = x_weight.repeat(1, 1, 2, fold // 2 // 2)
        x_weight = x_weight.view(x_weight.size(0), x_weight.size(1), -1)

        x_weight = x_weight[:, :, :, None, None]
        x_sa = x_sa * x_weight
        x_sa = x_sa.contiguous().view(nt, fold, h, w)

        out[:, :fold, :] = x_sa
        out[:, fold:, :] = x[:, fold:, :]
        return out


class CombinNet(nn.Module):
    def __init__(self, net1, net2):
        super(CombinNet, self).__init__()
        self.net1 = net1
        self.net2 = net2

    def forward(self, x):
        x = self.net1(x)
        x = self.net2(x)
        return x


def make_temporal_interlace(net, n_segment, place='blockres', shift_div=1):
    n_segment_list = [n_segment] * 4
    assert n_segment_list[-1] > 0
    logger.info('=> n_segment per stage: {}'.format(n_segment_list))

    import torchvision
    n_round = 1
    if len(list(net.layer3.children())) >= 23:
        logger.info('=> Using n_round {} to insert temporal shift'.format(n_round))

    def make_block_interlace(stage, this_segment, shift_div):
        blocks = list(stage.children())
        logger.info('=> Processing stage with {} blocks residual'.format(len(blocks)))
        for i, b in enumerate(blocks):
            if i % n_round == 0:
                tds = TemporalDeform(
                    b.conv1.in_channels,
                    n_segment=this_segment,
                    shift_div=shift_div)
                blocks[i].conv1 = CombinNet(tds, blocks[i].conv1)
        return nn.Sequential(*blocks)

    net.layer1 = make_block_interlace(net.layer1, n_segment_list[0], shift_div)
    net.layer2 = make_block_interlace(net.layer2, n_segment_list[1], shift_div)
    net.layer3 = make_block_interlace(net.layer3, n_segment_list[2], shift_div)
    net.layer4 = make_block_interlace(net.layer4, n_segment_list[3], shift_div)
