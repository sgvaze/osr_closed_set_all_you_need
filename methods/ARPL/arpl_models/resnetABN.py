import torch.nn as nn
import math
import torch
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from methods.ARPL.arpl_models.ABN import MultiBatchNorm
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _ntuple

from collections import OrderedDict
import operator
from itertools import islice

_pair = _ntuple(2)

__all__ = ['resnet18ABN', 'resnet34ABN', 'resnet50ABN', 'resnet101ABN', 'resnet152ABN']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1, kernel_size=3):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=1, bias=False)


# class Conv2d(nn.Conv2d):
#     def __init__(self, *args, **kwargs):
#         super(Conv2d, self).__init__(*args, **kwargs)
#
#     def forward(self, x, domain_label):
#         return F.conv2d(x, self.weight, self.bias, self.stride,
#                          self.padding, self.dilation, self.groups), domain_label

class Conv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode='zeros')

    def forward(self, input, domain_label):
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups), domain_label


class TwoInputSequential(nn.Module):
    r"""A sequential container forward with two inputs.
    """

    def __init__(self, *args):
        super(TwoInputSequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return TwoInputSequential(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx, module):
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    def __len__(self):
        return len(self._modules)

    def __dir__(self):
        keys = super(TwoInputSequential, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def forward(self, input1, input2):
        for module in self._modules.values():
            input1, input2 = module(input1, input2)
        return input1, input2

    
def resnet18ABN(num_classes=10, num_bns=2):
    model = ResNetABN(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, num_bns=num_bns)

    return model


def resnet34ABN(num_classes=10, num_bns=2):
    model = ResNetABN(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, num_bns=num_bns)

    return model

def resnet50ABN(num_classes=10, num_bns=2, first_layer_conv=3):
    model = ResNetABN(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, num_bns=num_bns,
                      first_layer_conv=first_layer_conv)

    return model





def _update_initial_weights_ABN(state_dict, num_classes=1000, num_bns=2, ABN_type='all'):
    new_state_dict = state_dict.copy()

    for key, val in state_dict.items():
        update_dict = False
        if ((('bn' in key or 'downsample.1' in key) and ABN_type == 'all') or
                (('bn1' in key) and ABN_type == 'partial-bn1')):
            update_dict = True

        if (update_dict):
            if 'weight' in key:
                for d in range(num_bns):
                    new_state_dict[key[0:-6] + 'bns.{}.weight'.format(d)] = val.data.clone()

            elif 'bias' in key:
                for d in range(num_bns):
                    new_state_dict[key[0:-4] + 'bns.{}.bias'.format(d)] = val.data.clone()

            if 'running_mean' in key:
                for d in range(num_bns):
                    new_state_dict[key[0:-12] + 'bns.{}.running_mean'.format(d)] = val.data.clone()

            if 'running_var' in key:
                for d in range(num_bns):
                    new_state_dict[key[0:-11] + 'bns.{}.running_var'.format(d)] = val.data.clone()

            if 'num_batches_tracked' in key:
                for d in range(num_bns):
                    new_state_dict[
                        key[0:-len('num_batches_tracked')] + 'bns.{}.num_batches_tracked'.format(d)] = val.data.clone()

    if num_classes != 1000 or len([key for key in new_state_dict.keys() if 'fc' in key]) > 1:
        key_list = list(new_state_dict.keys())
        for key in key_list:
            if 'fc' in key:
                print('pretrained {} are not used as initial params.'.format(key))
                del new_state_dict[key]

    return new_state_dict


class ResNetABN(nn.Module):
    def __init__(self, block, layers, num_classes=10, num_bns=2, first_layer_conv=3):
        self.inplanes = 64
        self.num_bns = num_bns
        self.num_classes = num_classes
        super(ResNetABN, self).__init__()
        self.conv1 = conv3x3(3, 64, kernel_size=first_layer_conv)
        self.bn1 = MultiBatchNorm(64, self.num_bns)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, num_bns=self.num_bns)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, num_bns=self.num_bns)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, num_bns=self.num_bns)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, num_bns=self.num_bns)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, num_bns=2):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = TwoInputSequential(
                Conv2d(self.inplanes, planes * block.expansion,
                       kernel_size=1, stride=stride, bias=False),
                MultiBatchNorm(planes * block.expansion, num_bns),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, num_bns=num_bns))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, num_bns=num_bns))

        return TwoInputSequential(*layers)

    def forward(self, x, rf=False, domain_label=None):
        if domain_label is None:
            domain_label = 0 * torch.ones(x.shape[0], dtype=torch.long).cuda()
        x = self.conv1(x)
        x, _ = self.bn1(x, domain_label)
        x = F.relu(x)
        x, _ = self.layer1(x, domain_label)
        x, _ = self.layer2(x, domain_label)
        x, _ = self.layer3(x, domain_label)
        x, _ = self.layer4(x, domain_label)

        x = self.avgpool(x)
        feat = x.view(x.size(0), -1)
        x = self.fc(feat)

        if rf:
            return feat, x
        else:
            return x

        
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, num_bns=2):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = MultiBatchNorm(planes, num_bns)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = MultiBatchNorm(planes, num_bns)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, domain_label):
        residual = x

        out = self.conv1(x)
        out, _ = self.bn1(out, domain_label)
        out = F.relu(out)

        out = self.conv2(out)
        out, _ = self.bn2(out, domain_label)

        if self.downsample is not None:
            residual, _ = self.downsample(x, domain_label)

        out += residual
        out = F.relu(out)

        return out, domain_label


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, num_bns=2):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = MultiBatchNorm(planes, num_bns)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = MultiBatchNorm(planes, num_bns)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = MultiBatchNorm(planes * 4, num_bns)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, domain_label):
        residual = x

        out = self.conv1(x)
        out, _ = self.bn1(out, domain_label)
        out = self.relu(out)

        out = self.conv2(out)
        out, _ = self.bn2(out, domain_label)
        out = self.relu(out)

        out = self.conv3(out)
        out, _ = self.bn3(out, domain_label)

        if self.downsample is not None:
            residual, _ = self.downsample(x, domain_label)

        out += residual
        out = self.relu(out)

        return out, domain_label