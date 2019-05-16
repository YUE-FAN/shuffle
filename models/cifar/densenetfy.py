# from torchvision.models import densenet121
# a = densenet121()
#
# print('    Total params: %.2fM' % (sum(p.numel() for p in a.parameters())/1000000.0))
#
# DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16))


import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


def densenet_1d(**kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    """
    model = DenseNet_1d(**kwargs)
    return model


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _DenseLayer_1d(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer_1d, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=1, stride=1, padding=0, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer_1d, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock_1d(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock_1d, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer_1d(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class _Transition_1d(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition_1d, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))


class DenseNet_1d(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """
    def __init__(self, layer, num_classes, growth_rate=32,
                 num_init_features=64, bn_size=4, drop_rate=0):

        super(DenseNet_1d, self).__init__()
        print('DenseNet_1d is used!')
        self.layer = layer

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
        ]))

        # Each denseblock
        num_features = num_init_features
        if layer == 0:
            self.features.add_module('pool', nn.AvgPool2d(kernel_size=32))
        if layer <= 0:
            block = _DenseBlock_1d(num_layers=6, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % 1, block)
            num_features = num_features + 6 * growth_rate
            trans = _Transition_1d(num_input_features=num_features, num_output_features=num_features // 2)
        else:
            block = _DenseBlock(num_layers=6, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % 1, block)
            num_features = num_features + 6 * growth_rate
            trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        self.features.add_module('transition%d' % 1, trans)
        num_features = num_features // 2

        if layer == 1:
            self.features.add_module('pool', nn.AvgPool2d(kernel_size=16))
        if layer <= 1:
            block = _DenseBlock_1d(num_layers=12, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % 2, block)
            num_features = num_features + 12 * growth_rate
            trans = _Transition_1d(num_input_features=num_features, num_output_features=num_features // 2)
        else:
            block = _DenseBlock(num_layers=12, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % 2, block)
            num_features = num_features + 12 * growth_rate
            trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        self.features.add_module('transition%d' % 2, trans)
        num_features = num_features // 2

        if layer == 2:
            self.features.add_module('pool', nn.AvgPool2d(kernel_size=8))
        if layer <= 2:
            block = _DenseBlock_1d(num_layers=24, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % 3, block)
            num_features = num_features + 24 * growth_rate
            trans = _Transition_1d(num_input_features=num_features, num_output_features=num_features // 2)
        else:
            block = _DenseBlock(num_layers=24, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % 3, block)
            num_features = num_features + 24 * growth_rate
            trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        self.features.add_module('transition%d' % 3, trans)
        num_features = num_features // 2

        if layer == 3:
            self.features.add_module('pool', nn.AvgPool2d(kernel_size=4))
        if layer <= 3:
            block = _DenseBlock_1d(num_layers=16, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % 4, block)
            num_features = num_features + 16 * growth_rate
            trans = _Transition_1d(num_input_features=num_features, num_output_features=num_features // 2)
        else:
            block = _DenseBlock(num_layers=16, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % 4, block)
            num_features = num_features + 16 * growth_rate
            trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        self.features.add_module('transition%d' % 4, trans)
        num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        if self.layer == 4:
            out = F.avg_pool2d(out, kernel_size=2, stride=1)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
