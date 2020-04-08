from torch import nn

from x_temporal.core.basic_ops import ConsensusModule
from x_temporal.core.transforms import *
from torch.nn.init import normal_, constant_
from x_temporal.models.resnet import *
from x_temporal.models.slowfast import *
import pretrainedmodels

import logging
logger = logging.getLogger('global')



class TSN(nn.Module):
    def __init__(self, num_class, num_segments, modality,
                 base_model='resnet101', new_length=None,
                 consensus_type='avg', before_softmax=True,
                 dropout=0.8, img_feature_dim=256,
                 crop_num=1, partial_bn=True, print_spec=True, pretrain=True,
                 is_shift=False, shift_div=8, shift_place='blockres',
                 temporal_pool=False, non_local=False, tin=False):
        super(TSN, self).__init__()
        self.modality = modality
        self.num_segments = num_segments
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        # the dimension of the CNN feature to represent each frame
        self.img_feature_dim = img_feature_dim
        self.pretrain = pretrain

        self.is_shift = is_shift
        self.shift_div = shift_div
        self.shift_place = shift_place


        self.tin = tin

        self.base_model_name = base_model
        self.temporal_pool = temporal_pool
        self.non_local = non_local

        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 5
        else:
            self.new_length = new_length
        if print_spec:
            logger.info(("""
    Initializing with base model: {}.
    Model Configurations:
          input_modality:     {}
          num_segments:       {}
          new_length:         {}
          consensus_module:   {}
          dropout_ratio:      {}
          img_feature_dim:    {}
            """.format(base_model, self.modality, self.num_segments, self.new_length, consensus_type, self.dropout, self.img_feature_dim)))

        self._prepare_base_model(base_model)

        feature_dim = self._prepare_tsn(num_class)

        if self.modality == 'Flow':
            logger.info("Converting the ImageNet model to a flow init model")
            self.base_model = self._construct_flow_model(self.base_model)
            logger.info("Done. Flow model ready...")
        elif self.modality == 'RGBDiff':
            logger.info("Converting the ImageNet model to RGB+Diff init model")
            self.base_model = self._construct_diff_model(self.base_model)
            logger.info("Done. RGBDiff model ready.")

        self.consensus = ConsensusModule(consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _prepare_tsn(self, num_class):
        feature_dim = getattr(self.base_model,
                              self.base_model.last_layer_name).in_features
        if self.dropout == 0:
            setattr(
                self.base_model,
                self.base_model.last_layer_name,
                nn.Linear(
                    feature_dim,
                    num_class))
            self.new_fc = None
        else:
            setattr(
                self.base_model,
                self.base_model.last_layer_name,
                nn.Dropout(
                    p=self.dropout))
            if self.consensus_type in ['TRN', 'TRNmultiscale']:
                self.new_fc = nn.Linear(feature_dim, self.img_feature_dim)
            else:
                self.new_fc = nn.Linear(feature_dim, num_class)

        std = 0.001
        if self.new_fc is None:
            normal_(
                getattr(
                    self.base_model,
                    self.base_model.last_layer_name).weight,
                0,
                std)
            constant_(
                getattr(
                    self.base_model,
                    self.base_model.last_layer_name).bias,
                0)
        else:
            if hasattr(self.new_fc, 'weight'):
                normal_(self.new_fc.weight, 0, std)
                constant_(self.new_fc.bias, 0)
        return feature_dim

    def _prepare_base_model(self, base_model, config={}):
        logger.info('=> base model: {}'.format(base_model))

        if base_model.startswith('resnet'):
            self.base_model = globals()[base_model](pretrained=self.pretrain)

            if self.is_shift:
                logger.info('Adding temporal shift...')
                from x_temporal.core.tsm import make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place)

            if self.tin:
                logger.info('Adding temporal interlace conv...')
                from x_temporal.core.tin import make_temporal_interlace
                make_temporal_interlace(
                    self.base_model,
                    self.num_segments,
                    shift_div=self.shift_div)

            if self.non_local:
                logger.info('Adding non-local module...')
                from x_temporal.core.non_local import make_non_local
                make_non_local(self.base_model, self.num_segments)

            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)

            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + \
                    [0] * 3 * self.new_length
                self.input_std = self.input_std + \
                    [np.mean(self.input_std) * 2] * 3 * self.new_length

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN, self).train(mode)
        count = 0
        if self._enable_pbn and mode:
            logger.info("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()
                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def forward(self, input, no_reshape=False):
        if not no_reshape:
            sample_len = (3 if self.modality == "RGB" else 2) * self.new_length

            if self.modality == 'RGBDiff':
                sample_len = 3 * self.new_length
                input = self._get_diff(input)

            base_out = self.base_model(input.view(
                (-1, sample_len) + input.size()[-2:]))
        else:
            base_out = self.base_model(input)

        if self.dropout > 0:
            base_out = self.new_fc(base_out)

        if not self.before_softmax:
            base_out = self.softmax(base_out)

        if self.reshape:
            if self.is_shift and self.temporal_pool:
                base_out = base_out.view(
                    (-1, self.num_segments // 2) + base_out.size()[1:])
            else:
                base_out = base_out.view(
                    (-1, self.num_segments) + base_out.size()[1:])
            output = self.consensus(base_out)
            return output.squeeze(1)

    def _get_diff(self, input, keep_rgb=False):
        input_c = 3 if self.modality in ["RGB", "RGBDiff"] else 2
        input_view = input.view(
            (-1, self.num_segments, self.new_length + 1, input_c,) + input.size()[2:])
        if keep_rgb:
            new_data = input_view.clone()
        else:
            new_data = input_view[:, :, 1:, :, :, :].clone()

        for x in reversed(list(range(1, self.new_length + 1))):
            if keep_rgb:
                new_data[:, :, x, :, :, :] = input_view[:, :, x,
                                                        :, :, :] - input_view[:, :, x - 1, :, :, :]
            else:
                new_data[:, :, x - 1, :, :, :] = input_view[:, :,
                                                            x, :, :, :] - input_view[:, :, x - 1, :, :, :]

        return new_data

    def _construct_flow_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = list(
            filter(
                lambda x: isinstance(
                    modules[x], nn.Conv2d), list(
                    range(
                        len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution
        # kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + \
            (2 * self.new_length, ) + kernel_size[2:]
        new_kernels = params[0].data.mean(
            dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(2 * self.new_length, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        # remove .weight suffix to get the layer name
        layer_name = list(container.state_dict().keys())[0][:-7]

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)

        if self.base_model_name == 'BNInception':
            import torch.utils.model_zoo as model_zoo
            sd = model_zoo.load_url(
                'https://www.dropbox.com/s/35ftw2t4mxxgjae/BNInceptionFlow-ef652051.pth.tar?dl=1')
            base_model.load_state_dict(sd)
            logger.info('=> Loading pretrained Flow weight done...')
        else:
            logger.info('#' * 30, 'Warning! No Flow pretrained model is found')
        return base_model

    def _construct_diff_model(self, base_model, keep_rgb=False):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = filter(
            lambda x: isinstance(
                modules[x], nn.Conv2d), list(
                range(
                    len(modules))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution
        # kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        if not keep_rgb:
            new_kernel_size = kernel_size[:1] + \
                (3 * self.new_length,) + kernel_size[2:]
            new_kernels = params[0].data.mean(
                dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        else:
            new_kernel_size = kernel_size[:1] + \
                (3 * self.new_length,) + kernel_size[2:]
            new_kernels = torch.cat((params[0].data, params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()),
                                    1)
            new_kernel_size = kernel_size[:1] + \
                (3 + 3 * self.new_length,) + kernel_size[2:]

        new_conv = nn.Conv2d(new_kernel_size[1], conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        # remove .weight suffix to get the layer name
        layer_name = list(container.state_dict().keys())[0][:-7]

        # replace the first convolution layer
        setattr(container, layer_name, new_conv)
        return base_model

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self, flip=True):
        if self.modality == 'RGB':
            if flip:
                return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                       GroupRandomHorizontalFlip(is_flow=False)])
            else:
                logger.info('#' * 20, 'NO FLIP!!!')
                return torchvision.transforms.Compose(
                    [GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66])])
        elif self.modality == 'Flow':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=True)])
        elif self.modality == 'RGBDiff':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
