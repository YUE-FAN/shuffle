import torch.nn as nn
import math
# import torch
# import numpy as np
# from .dconv import DConv1Dai_share, DConv1Dai, Dconv_cos, Dconv_euc, Dconv_rand, Dconv_drop, Dconv_shuffle, Dconv_shuffleall, Dconv_none, Dconv_horizontal, Dconv_vertical
# from .dconv import Dconv_cshuffle, Dconv_crand, Dconv_localshuffle


class CONV_3x3(nn.Module):
    def __init__(self, inplanes, outplanes, stride):
        super(CONV_3x3, self).__init__()
        self.conv = nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(outplanes, eps=0.001, momentum=1-0.9997)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CONV_DW_3x3(nn.Module):
    def __init__(self, inplanes, outplanes, stride):
        super(CONV_DW_3x3, self).__init__()
        self.conv_dw = nn.Conv2d(inplanes, inplanes, 3, stride, 1, groups=inplanes, bias=False)
        self.bn_dw = nn.BatchNorm2d(inplanes, eps=0.001, momentum=1-0.9997)
        self.conv_1x1 = nn.Conv2d(inplanes, outplanes, 1, 1, 0, bias=False)
        self.bn_1x1 = nn.BatchNorm2d(outplanes, eps=0.001, momentum=1-0.9997)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_dw(x)
        x = self.bn_dw(x)
        x = self.relu(x)
        x = self.conv_1x1(x)
        x = self.bn_1x1(x)
        x = self.relu(x)
        return x


class CONV1D_3x3(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(CONV1D_3x3, self).__init__()
        self.conv1d = nn.Linear(inplanes, outplanes, bias=False)
        self.bn = nn.BatchNorm1d(outplanes, eps=0.001, momentum=1-0.9997)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1d(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CONV1D_DW_3x3(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(CONV1D_DW_3x3, self).__init__()
        self.conv_dw = nn.Linear(inplanes, inplanes, bias=False)
        self.bn_dw = nn.BatchNorm1d(inplanes, eps=0.001, momentum=1-0.9997)
        self.conv_1x1 = nn.Linear(inplanes, outplanes, bias=False)
        self.bn_1x1 = nn.BatchNorm1d(outplanes, eps=0.001, momentum=1-0.9997)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_dw(x)
        x = self.bn_dw(x)
        x = self.relu(x)
        x = self.conv_1x1(x)
        x = self.bn_1x1(x)
        x = self.relu(x)
        return x


class CONV_1x1(nn.Module):
    def __init__(self, inplanes, outplanes, stride):
        super(CONV_1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(outplanes, eps=0.001, momentum=1-0.9997)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CONV_DW_1x1(nn.Module):
    def __init__(self, inplanes, outplanes, stride):
        super(CONV_DW_1x1, self).__init__()
        # self.conv_dw = nn.Conv2d(inplanes, inplanes, 1, stride, 0, groups=inplanes, bias=False)
        # self.bn_dw = nn.BatchNorm2d(inplanes, eps=0.001, momentum=1-0.9997)
        self.conv_1x1 = nn.Conv2d(inplanes, outplanes, 1, 1, 0, bias=False)
        self.bn_1x1 = nn.BatchNorm2d(outplanes, eps=0.001, momentum=1-0.9997)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x = self.conv_dw(x)
        # x = self.bn_dw(x)
        # x = self.relu(x)
        x = self.conv_1x1(x)
        x = self.bn_1x1(x)
        x = self.relu(x)
        return x

def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class InvertedResidual1x1(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual1x1, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 1, stride, 0, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 1, stride, 0, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV1_1d(nn.Module):
    def __init__(self, dropout_rate, num_classes, include_top, layer):
        super(MobileNetV1_1d, self).__init__()
        print("CIFAR MobileNetV1_1d is used")
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.include_top = include_top
        self.layer = layer

        if layer > 11:
            self.conv11 = CONV_3x3(3, 32, 2)
        else:
            self.conv11 = CONV1D_3x3(3, 32)
        if layer > 12:
            self.conv12 = CONV_DW_3x3(32, 64, 1)
        else:
            self.conv12 = CONV1D_DW_3x3(32, 64)

        if layer > 21:
            self.conv21 = CONV_DW_3x3(64, 128, 2)
        else:
            self.conv21 = CONV1D_DW_3x3(64, 128)
        if layer > 22:
            self.conv22 = CONV_DW_3x3(128, 128, 1)
        else:
            self.conv22 = CONV1D_DW_3x3(128, 128)

        if layer > 31:
            self.conv31 = CONV_DW_3x3(128, 256, 2)
        else:
            self.conv31 = CONV1D_DW_3x3(128, 256)
        if layer > 32:
            self.conv32 = CONV_DW_3x3(256, 256, 1)
        else:
            self.conv32 = CONV1D_DW_3x3(256, 256)

        if layer > 41:
            self.conv41 = CONV_DW_3x3(256, 512, 2)
        else:
            self.conv41 = CONV1D_DW_3x3(256, 512)
        if layer > 42:
            self.conv42 = CONV_DW_3x3(512, 512, 1)
        else:
            self.conv42 = CONV1D_DW_3x3(512, 512)
        if layer > 43:
            self.conv43 = CONV_DW_3x3(512, 512, 1)
        else:
            self.conv43 = CONV1D_DW_3x3(512, 512)
        if layer > 44:
            self.conv44 = CONV_DW_3x3(512, 512, 1)
        else:
            self.conv44 = CONV1D_DW_3x3(512, 512)
        if layer > 45:
            self.conv45 = CONV_DW_3x3(512, 512, 1)
        else:
            self.conv45 = CONV1D_DW_3x3(512, 512)
        if layer > 46:
            self.conv46 = CONV_DW_3x3(512, 512, 1)
        else:
            self.conv46 = CONV1D_DW_3x3(512, 512)

        if layer > 51:
            self.conv51 = CONV_DW_3x3(512, 1024, 2)
        else:
            self.conv51 = CONV1D_DW_3x3(512, 1024)
        if layer > 52:
            self.conv52 = CONV_DW_3x3(1024, 1024, 1)
        else:
            self.conv52 = CONV1D_DW_3x3(1024, 1024)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)

        # Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                # raise Exception('You are using a model without BN!!!')
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                # raise Exception('You are using a model without BN!!!')

    def forward(self, x):
        if self.layer == 11:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        x = self.conv11(x)
        if self.layer == 12:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        x = self.conv12(x)
        if self.layer == 21:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        x = self.conv21(x)
        if self.layer == 22:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        x = self.conv22(x)
        if self.layer == 31:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        x = self.conv31(x)
        if self.layer == 32:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        x = self.conv32(x)
        if self.layer == 41:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        x = self.conv41(x)
        if self.layer == 42:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        x = self.conv42(x)
        if self.layer == 43:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        x = self.conv43(x)
        if self.layer == 44:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        x = self.conv44(x)
        if self.layer == 45:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        x = self.conv45(x)
        if self.layer == 46:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        x = self.conv46(x)
        if self.layer == 51:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        x = self.conv51(x)
        if self.layer == 52:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        x = self.conv52(x)
        # print("feature shape:", x.size())
        if self.layer == 99:
            x = self.avgpool(x)

        if self.include_top:
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        return x


class MobileNetV1_1x1(nn.Module):
    def __init__(self, dropout_rate, num_classes, include_top, layer):
        super(MobileNetV1_1x1, self).__init__()
        print("CIFAR MobileNetV1_1x1 is used")
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.include_top = include_top
        self.layer = layer

        if layer > 11:
            self.conv11 = CONV_3x3(3, 32, 2)
        else:
            self.conv11 = CONV_1x1(3, 32, 2)
        if layer > 12:
            self.conv12 = CONV_DW_3x3(32, 64, 1)
        else:
            self.conv12 = CONV_DW_1x1(32, 64, 1)

        if layer > 21:
            self.conv21 = CONV_DW_3x3(64, 128, 2)
        else:
            self.conv21 = CONV_DW_1x1(64, 128, 2)
        if layer > 22:
            self.conv22 = CONV_DW_3x3(128, 128, 1)
        else:
            self.conv22 = CONV_DW_1x1(128, 128, 1)

        if layer > 31:
            self.conv31 = CONV_DW_3x3(128, 256, 2)
        else:
            self.conv31 = CONV_DW_1x1(128, 256, 2)
        if layer > 32:
            self.conv32 = CONV_DW_3x3(256, 256, 1)
        else:
            self.conv32 = CONV_DW_1x1(256, 256, 1)

        if layer > 41:
            self.conv41 = CONV_DW_3x3(256, 512, 2)
        else:
            self.conv41 = CONV_DW_1x1(256, 512, 2)
        if layer > 42:
            self.conv42 = CONV_DW_3x3(512, 512, 1)
        else:
            self.conv42 = CONV_DW_1x1(512, 512, 1)
        if layer > 43:
            self.conv43 = CONV_DW_3x3(512, 512, 1)
        else:
            self.conv43 = CONV_DW_1x1(512, 512, 1)
        if layer > 44:
            self.conv44 = CONV_DW_3x3(512, 512, 1)
        else:
            self.conv44 = CONV_DW_1x1(512, 512, 1)
        if layer > 45:
            self.conv45 = CONV_DW_3x3(512, 512, 1)
        else:
            self.conv45 = CONV_DW_1x1(512, 512, 1)
        if layer > 46:
            self.conv46 = CONV_DW_3x3(512, 512, 1)
        else:
            self.conv46 = CONV_DW_1x1(512, 512, 1)

        if layer > 51:
            self.conv51 = CONV_DW_3x3(512, 1024, 2)
        else:
            self.conv51 = CONV_DW_1x1(512, 1024, 2)
        if layer > 52:
            self.conv52 = CONV_DW_3x3(1024, 1024, 1)
        else:
            self.conv52 = CONV_DW_1x1(1024, 1024, 1)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(1024, num_classes)

        # Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                # raise Exception('You are using a model without BN!!!')
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                # raise Exception('You are using a model without BN!!!')

    def forward(self, x):
        x = self.conv11(x)
        x = self.conv12(x)

        x = self.conv21(x)
        x = self.conv22(x)

        x = self.conv31(x)
        x = self.conv32(x)

        x = self.conv41(x)
        x = self.conv42(x)
        x = self.conv43(x)
        x = self.conv44(x)
        x = self.conv45(x)
        x = self.conv46(x)

        x = self.conv51(x)
        x = self.conv52(x)

        if self.include_top:
            x = self.avgpool(x)
            x = self.dropout(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        return x


class MobileNetV1_truncated(nn.Module):
    def __init__(self, dropout_rate, num_classes, include_top, layer):
        super(MobileNetV1_truncated, self).__init__()
        print("CIFAR MobileNetV1_truncated is used")
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.include_top = include_top
        self.layer = layer
        self.modulelist = nn.ModuleList()

        if layer > 11:
            self.modulelist.append(CONV_3x3(3, 32, 2))
        if layer > 12:
            self.modulelist.append(CONV_DW_3x3(32, 64, 1))

        if layer > 21:
            self.modulelist.append(CONV_DW_3x3(64, 128, 2))
        if layer > 22:
            self.modulelist.append(CONV_DW_3x3(128, 128, 1))

        if layer > 31:
            self.modulelist.append(CONV_DW_3x3(128, 256, 2))
        if layer > 32:
            self.modulelist.append(CONV_DW_3x3(256, 256, 1))

        if layer > 41:
            self.modulelist.append(CONV_DW_3x3(256, 512, 2))
        if layer > 42:
            self.modulelist.append(CONV_DW_3x3(512, 512, 1))
        if layer > 43:
            self.modulelist.append(CONV_DW_3x3(512, 512, 1))
        if layer > 44:
            self.modulelist.append(CONV_DW_3x3(512, 512, 1))
        if layer > 45:
            self.modulelist.append(CONV_DW_3x3(512, 512, 1))
        if layer > 46:
            self.modulelist.append(CONV_DW_3x3(512, 512, 1))

        if layer > 51:
            self.modulelist.append(CONV_DW_3x3(512, 1024, 2))
        if layer > 52:
            self.modulelist.append(CONV_DW_3x3(1024, 1024, 1))

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        if layer == 11:
            s = 32
        elif layer == 12:
            s = 64
        elif layer in [21, 22, 31, 32, 41, 42, 43, 44, 45, 46, 51, 52, 99]:
            s = self.modulelist[-1].conv_1x1.out_channels
        self.fc = nn.Linear(s, num_classes)
        print(len(self.modulelist))

        # Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                raise Exception('You are using a model without BN!!!')

    def forward(self, x):
        for i, module in enumerate(self.modulelist):
            # print("module %d has input feature shape:" % i, x.size())
            x = module(x)

        if self.include_top:
            x = self.avgpool(x)
            x = self.dropout(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        return x


class MobileNetV1_1x1LMP(nn.Module):
    def __init__(self, dropout_rate, num_classes, include_top, layer):
        super(MobileNetV1_1x1LMP, self).__init__()
        print("CIFAR MobileNetV1_1x1LMP is used")
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.include_top = include_top
        self.layer = layer

        possible_layers = [11, 12, 21, 22, 31, 32, 41, 42, 43, 44, 45, 46, 51, 52, 99]
        if layer not in possible_layers:
            raise Exception('the layer you choose for MobileNetV1_1x1LMP is invaild!!!')

        if layer > 11:
            self.conv11 = CONV_3x3(3, 32, 2)
        else:
            self.conv11 = nn.Sequential(CONV_1x1(3, 32, 1), nn.MaxPool2d(2, 2))
        if layer > 12:
            self.conv12 = CONV_DW_3x3(32, 64, 1)
        else:
            self.conv12 = CONV_DW_1x1(32, 64, 1)

        if layer > 21:
            self.conv21 = CONV_DW_3x3(64, 128, 2)
        else:
            self.conv21 = nn.Sequential(CONV_DW_1x1(64, 128, 1), nn.MaxPool2d(2, 2))
        if layer > 22:
            self.conv22 = CONV_DW_3x3(128, 128, 1)
        else:
            self.conv22 = CONV_DW_1x1(128, 128, 1)

        if layer > 31:
            self.conv31 = CONV_DW_3x3(128, 256, 2)
        else:
            self.conv31 = nn.Sequential(CONV_DW_1x1(128, 256, 1), nn.MaxPool2d(2, 2))
        if layer > 32:
            self.conv32 = CONV_DW_3x3(256, 256, 1)
        else:
            self.conv32 = CONV_DW_1x1(256, 256, 1)

        if layer > 41:
            self.conv41 = CONV_DW_3x3(256, 512, 2)
        else:
            self.conv41 = nn.Sequential(CONV_DW_1x1(256, 512, 1), nn.MaxPool2d(2, 2))
        if layer > 42:
            self.conv42 = CONV_DW_3x3(512, 512, 1)
        else:
            self.conv42 = CONV_DW_1x1(512, 512, 1)
        if layer > 43:
            self.conv43 = CONV_DW_3x3(512, 512, 1)
        else:
            self.conv43 = CONV_DW_1x1(512, 512, 1)
        if layer > 44:
            self.conv44 = CONV_DW_3x3(512, 512, 1)
        else:
            self.conv44 = CONV_DW_1x1(512, 512, 1)
        if layer > 45:
            self.conv45 = CONV_DW_3x3(512, 512, 1)
        else:
            self.conv45 = CONV_DW_1x1(512, 512, 1)
        if layer > 46:
            self.conv46 = CONV_DW_3x3(512, 512, 1)
        else:
            self.conv46 = CONV_DW_1x1(512, 512, 1)

        if layer > 51:
            self.conv51 = CONV_DW_3x3(512, 1024, 2)
        else:
            self.conv51 = nn.Sequential(CONV_DW_1x1(512, 1024, 1), nn.MaxPool2d(2, 2))
        if layer > 52:
            self.conv52 = CONV_DW_3x3(1024, 1024, 1)
        else:
            self.conv52 = CONV_DW_1x1(1024, 1024, 1)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(1024, num_classes)

        # Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                # raise Exception('You are using a model without BN!!!')
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                # raise Exception('You are using a model without BN!!!')

    def forward(self, x):
        x = self.conv11(x)
        x = self.conv12(x)

        x = self.conv21(x)
        x = self.conv22(x)

        x = self.conv31(x)
        x = self.conv32(x)

        x = self.conv41(x)
        x = self.conv42(x)
        x = self.conv43(x)
        x = self.conv44(x)
        x = self.conv45(x)
        x = self.conv46(x)

        x = self.conv51(x)
        x = self.conv52(x)

        if self.include_top:
            x = self.avgpool(x)
            x = self.dropout(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        return x


class MobileNetV1_1x1LAP(nn.Module):
    def __init__(self, dropout_rate, num_classes, include_top, layer):
        super(MobileNetV1_1x1LAP, self).__init__()
        print("CIFAR MobileNetV1_1x1LAP is used")
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.include_top = include_top
        self.layer = layer

        possible_layers = [11, 12, 21, 22, 31, 32, 41, 42, 43, 44, 45, 46, 51, 52, 99]
        if layer not in possible_layers:
            raise Exception('the layer you choose for MobileNetV1_1x1LMP is invaild!!!')

        if layer > 11:
            self.conv11 = CONV_3x3(3, 32, 2)
        else:
            self.conv11 = nn.Sequential(CONV_1x1(3, 32, 1), nn.AvgPool2d(2, 2))
        if layer > 12:
            self.conv12 = CONV_DW_3x3(32, 64, 1)
        else:
            self.conv12 = CONV_DW_1x1(32, 64, 1)

        if layer > 21:
            self.conv21 = CONV_DW_3x3(64, 128, 2)
        else:
            self.conv21 = nn.Sequential(CONV_DW_1x1(64, 128, 1), nn.AvgPool2d(2, 2))
        if layer > 22:
            self.conv22 = CONV_DW_3x3(128, 128, 1)
        else:
            self.conv22 = CONV_DW_1x1(128, 128, 1)

        if layer > 31:
            self.conv31 = CONV_DW_3x3(128, 256, 2)
        else:
            self.conv31 = nn.Sequential(CONV_DW_1x1(128, 256, 1), nn.AvgPool2d(2, 2))
        if layer > 32:
            self.conv32 = CONV_DW_3x3(256, 256, 1)
        else:
            self.conv32 = CONV_DW_1x1(256, 256, 1)

        if layer > 41:
            self.conv41 = CONV_DW_3x3(256, 512, 2)
        else:
            self.conv41 = nn.Sequential(CONV_DW_1x1(256, 512, 1), nn.AvgPool2d(2, 2))
        if layer > 42:
            self.conv42 = CONV_DW_3x3(512, 512, 1)
        else:
            self.conv42 = CONV_DW_1x1(512, 512, 1)
        if layer > 43:
            self.conv43 = CONV_DW_3x3(512, 512, 1)
        else:
            self.conv43 = CONV_DW_1x1(512, 512, 1)
        if layer > 44:
            self.conv44 = CONV_DW_3x3(512, 512, 1)
        else:
            self.conv44 = CONV_DW_1x1(512, 512, 1)
        if layer > 45:
            self.conv45 = CONV_DW_3x3(512, 512, 1)
        else:
            self.conv45 = CONV_DW_1x1(512, 512, 1)
        if layer > 46:
            self.conv46 = CONV_DW_3x3(512, 512, 1)
        else:
            self.conv46 = CONV_DW_1x1(512, 512, 1)

        if layer > 51:
            self.conv51 = CONV_DW_3x3(512, 1024, 2)
        else:
            self.conv51 = nn.Sequential(CONV_DW_1x1(512, 1024, 1), nn.AvgPool2d(2, 2))
        if layer > 52:
            self.conv52 = CONV_DW_3x3(1024, 1024, 1)
        else:
            self.conv52 = CONV_DW_1x1(1024, 1024, 1)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(1024, num_classes)

        # Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                # raise Exception('You are using a model without BN!!!')
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                # raise Exception('You are using a model without BN!!!')

    def forward(self, x):
        x = self.conv11(x)
        x = self.conv12(x)

        x = self.conv21(x)
        x = self.conv22(x)

        x = self.conv31(x)
        x = self.conv32(x)

        x = self.conv41(x)
        x = self.conv42(x)
        x = self.conv43(x)
        x = self.conv44(x)
        x = self.conv45(x)
        x = self.conv46(x)

        x = self.conv51(x)
        x = self.conv52(x)

        if self.include_top:
            x = self.avgpool(x)
            x = self.dropout(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        return x


class MobileNetV2_1x1(nn.Module):
    def __init__(self, num_classes, layer):
        super(MobileNetV2_1x1, self).__init__()
        print("MobileNetV2_1x1 is used")
        block = InvertedResidual
        block1x1 = InvertedResidual1x1

        assert layer in [10, 20, 21, 30, 31, 32, 40, 41, 42, 43, 50, 51, 52, 60, 61, 62, 70, 99]
        layers = [conv_3x3_bn(3, 32, 2)]

        if layer > 10:
            layers.append(block(32, 16, 1, 1))
        else:
            layers.append(block1x1(32, 16, 1, 1))

        if layer > 20:
            layers.append(block(16, 24, 2, 6))
        else:
            layers.append(block1x1(16, 24, 2, 6))
        if layer > 21:
            layers.append(block(24, 24, 1, 6))
        else:
            layers.append(block1x1(24, 24, 1, 6))

        if layer > 30:
            layers.append(block(24, 32, 2, 6))
        else:
            layers.append(block1x1(24, 32, 2, 6))
        if layer > 31:
            layers.append(block(32, 32, 1, 6))
        else:
            layers.append(block1x1(32, 32, 1, 6))
        if layer > 32:
            layers.append(block(32, 32, 1, 6))
        else:
            layers.append(block1x1(32, 32, 1, 6))

        if layer > 40:
            layers.append(block(32, 64, 2, 6))
        else:
            layers.append(block1x1(32, 64, 2, 6))
        if layer > 41:
            layers.append(block(64, 64, 1, 6))
        else:
            layers.append(block1x1(64, 64, 1, 6))
        if layer > 42:
            layers.append(block(64, 64, 1, 6))
        else:
            layers.append(block1x1(64, 64, 1, 6))
        if layer > 43:
            layers.append(block(64, 64, 1, 6))
        else:
            layers.append(block1x1(64, 64, 1, 6))

        if layer > 50:
            layers.append(block(64, 96, 1, 6))
        else:
            layers.append(block1x1(64, 96, 1, 6))
        if layer > 51:
            layers.append(block(96, 96, 1, 6))
        else:
            layers.append(block1x1(96, 96, 1, 6))
        if layer > 52:
            layers.append(block(96, 96, 1, 6))
        else:
            layers.append(block1x1(96, 96, 1, 6))

        if layer > 60:
            layers.append(block(96, 160, 2, 6))
        else:
            layers.append(block1x1(96, 160, 2, 6))
        if layer > 61:
            layers.append(block(160, 160, 1, 6))
        else:
            layers.append(block1x1(160, 160, 1, 6))
        if layer > 62:
            layers.append(block(160, 160, 1, 6))
        else:
            layers.append(block1x1(160, 160, 1, 6))

        if layer > 70:
            layers.append(block(160, 320, 1, 6))
        else:
            layers.append(block1x1(160, 320, 1, 6))

        self.features = nn.Sequential(*layers)

        self.conv = conv_1x1_bn(320, 1280)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.classifier = nn.Linear(1280, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenetv2_1x1(**kwargs):
    """
    Constructs a MobileNetV2_1x1 model
    """
    return MobileNetV2_1x1(**kwargs)


def mobilenetv1_1d(**kwargs):
    """
    Constructs a MobileNetV1_1d model.
    """
    return MobileNetV1_1d(**kwargs)


def mobilenetv1_1x1(**kwargs):
    """
    Constructs a MobileNetV1_1x1 model.
    """
    return MobileNetV1_1x1(**kwargs)


def mobilenetv1_truncated(**kwargs):
    """
    Constructs a MobileNetV1_truncated model.
    """
    return MobileNetV1_truncated(**kwargs)


def mobilenetv1_1x1lmp(**kwargs):
    """
    Constructs a MobileNetV1_1x1LMP model.
    """
    return MobileNetV1_1x1LMP(**kwargs)


def mobilenetv1_1x1lap(**kwargs):
    """
    Constructs a MobileNetV1_1x1LAP model.
    """
    return MobileNetV1_1x1LAP(**kwargs)
