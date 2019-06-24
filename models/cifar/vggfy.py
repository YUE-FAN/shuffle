import torch.nn as nn
import torch
import numpy as np
from .dconv import DConv1Dai_share, DConv1Dai, Dconv_cos, Dconv_euc, Dconv_rand, Dconv_drop, Dconv_shuffle, Dconv_shuffleall, Dconv_none, Dconv_horizontal, Dconv_vertical
from .dconv import Dconv_cshuffle, Dconv_crand, Dconv_localshuffle


class CONV_3x3(nn.Module):
    """
    This is just a wraper for a conv3x3
    """
    def __init__(self, inplanes, outplanes, kernelsize, stride, padding, bias):
        super(CONV_3x3, self).__init__()
        self.outchannels = outplanes
        if padding == 'same':
            p = int((kernelsize - 1) / 2)
        elif padding == 'valid':
            p = 0
        else:
            raise Exception('padding should be either same or valid')
        self.conv = nn.Conv2d(inplanes, outplanes, kernel_size=kernelsize, stride=stride, padding=p, bias=bias)
        self.bn = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # np.save('/nethome/yuefan/fanyue/dconv/cifar100_weights/weight'+str(n)+'.npy', self.conv.weight.detach().cpu().numpy())
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    
class CONV_1x1(nn.Module):
    """
    This is just a wraper for a CONV_1x1
    """
    def __init__(self, inplanes, outplanes, stride, padding, bias):
        super(CONV_1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CONV1Dshuff_3x3(nn.Module):
    """
    In order to show that spatial relation is not important, I do GAP after 22 layer,
    then I just apply conv1D onto the channels until the very end
    """
    def __init__(self, inplanes, outplanes, bias=False):
        super(CONV1Dshuff_3x3, self).__init__()
        self.conv1d = nn.Linear(inplanes, outplanes, bias=bias)
        self.bn = nn.BatchNorm1d(outplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_shape = x.size()
        x_shuff = torch.empty(x_shape[0], x_shape[1]).cuda(0)
        perm = torch.randperm(x_shape[1])
        x_shuff[:, :] = x[:, perm]
        x = self.conv1d(x_shuff)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CONV1D_3x3(nn.Module):
    """
    In order to show that spatial relation is not important, I do GAP after 22 layer,
    then I just apply conv1D onto the channels until the very end
    """
    def __init__(self, inplanes, outplanes, bias=False):
        super(CONV1D_3x3, self).__init__()
        self.conv1d = nn.Linear(inplanes, outplanes, bias=bias)
        self.bn = nn.BatchNorm1d(outplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1d(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CONV3D_3x3(nn.Module):
    """
    2D kernels should also slide along the channel, but it is implemented in CONV3d
    """
    def __init__(self, inplanes, outplanes, kernelsize, stride, padding, bias):
        kernelsize_depth = 51  # TODO: tune thie parameter!!!!!
        super(CONV3D_3x3, self).__init__()
        if padding == 'same':
            p = (kernelsize - 1) / 2
            p_depth = (kernelsize_depth - 1) / 2
        elif padding == 'valid':
            p = 0
            p_depth = 0
        else:
            raise Exception('padding should be either same or valid')
        self.conv3d = nn.Conv3d(1, outplanes, kernel_size=(kernelsize_depth, kernelsize, kernelsize),
                                stride=stride, padding=(p_depth, p, p), bias=bias)
        self.bn = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        xshape = x.size()
        x = x.view(xshape[0], 1, xshape[1], xshape[2], xshape[3])
        x = self.conv3d(x)
        x = torch.mean(x, dim=2, keepdim=False)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CONV_3x3_save(nn.Module):
    """
    This is just a wraper for a conv3x3
    """
    def __init__(self, inplanes, outplanes, kernelsize, stride, padding, bias):
        super(CONV_3x3_save, self).__init__()
        if padding == 'same':
            p = (kernelsize - 1) / 2
        elif padding == 'valid':
            p = 0
        else:
            raise Exception('padding should be either same or valid')
        self.conv = nn.Conv2d(inplanes, outplanes, kernel_size=kernelsize, stride=stride, padding=p, bias=bias)
        self.bn = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # print(self.conv.weight.size())
        np.save('/nethome/yuefan/fanyue/dconv/weight13.npy', self.conv.weight.detach().cpu().numpy())
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DCONV_3x3(nn.Module):
    """
    This is just a wraper for a dconv3x3
    """

    def __init__(self, inplanes, outplanes, kernelsize, stride, padding, nrows, ncols, type):
        super(DCONV_3x3, self).__init__()
        """
        This if-elif is only for shuffle experiments, remember to get rid of it afterwards!
        """
        # if type=='none':
        #     self.dconv = Dconv_none(inplanes, outplanes, kernelsize, stride, padding)
        # elif type=='shuffleall':
        #     self.dconv = Dconv_shuffleall(inplanes, outplanes, kernelsize, stride, padding)
        # elif type=='shuffle':
        #     self.dconv = Dconv_shuffle(inplanes, outplanes, kernelsize, stride, padding)
        # elif type=='cshuffle':
        #     self.dconv = Dconv_cshuffle(inplanes, outplanes, kernelsize, stride, padding)
        # elif type=='rand':
        #     self.dconv = Dconv_rand(inplanes, outplanes, kernelsize, stride, padding)
        # elif type=='crand':
        #     self.dconv = Dconv_crand(inplanes, outplanes, kernelsize, stride, padding)
        # else:
        #     raise Exception('The type of the dconv does not exit')
        self.dconv = Dconv_localshuffle(inplanes, outplanes, kernelsize, stride, padding, nrows, ncols)
        self.bn = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # np.save('/nethome/yuefan/fanyue/dconv/weight52.npy', self.dconv.dilated_conv.weight.detach().cpu().numpy())
        # np.save('/nethome/yuefan/fanyue/dconv/x53.npy', x.detach().cpu().numpy())
        x = self.dconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


def SpatialAttn_whr(x):
    """Spatial Attention"""
    x_shape = x.size()
    a = x.sum(1, keepdim=True)
    a = a.view(x_shape[0], -1)
    a = a / a.sum(1, keepdim=True)
    a = a.view(x_shape[0], 1, x_shape[2], x_shape[3])
    return a


def ChannelAttn_whr(x):
    """Channel Attention"""
    x_shape = x.size()
    x = x.view(x_shape[0], x_shape[1], -1)  # [bs, c, h*w]
    a = x.sum(-1, keepdim=False)  # [bs, c]
    a = a / a.sum(1, keepdim=True)  # [bs, c]
    a = a.unsqueeze(-1).unsqueeze(-1)
    return a


class VGG19(nn.Module):  # TODO: try different config of the channels
    def __init__(self, dropout_rate, num_classes, include_top):
        super(VGG19, self).__init__()
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.include_top = include_top

        # Define the building blocks
        self.conv11 = CONV_3x3(3, 64, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv12 = CONV_3x3(64, 64, kernelsize=3, stride=2, padding='same', bias=False)

        self.conv21 = CONV_3x3(64, 128, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv22 = CONV_3x3(128, 128, kernelsize=3, stride=2, padding='same', bias=False)

        self.conv31 = CONV_3x3(128, 256, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv32 = CONV_3x3(256, 256, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv33 = CONV_3x3(256, 256, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv34 = CONV_3x3(256, 256, kernelsize=3, stride=2, padding='same', bias=False)

        self.conv41 = CONV_3x3(256, 512, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv42 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv43 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv44 = CONV_3x3(512, 512, kernelsize=3, stride=2, padding='same', bias=False)

        self.conv51 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv52 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv53 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv54 = CONV_3x3(512, 512, kernelsize=3, stride=2, padding='same', bias=False)

        self.fc = nn.Linear(512, num_classes)

        # Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_x):
        # print('input size:', input_x.size())
        x = self.conv11(input_x)
        x = self.conv12(x)

        x = self.conv21(x)
        x = self.conv22(x)

        x = self.conv31(x)
        x = self.conv32(x)
        x = self.conv33(x)
        x = self.conv34(x)

        x = self.conv41(x)
        x = self.conv42(x)
        x = self.conv43(x)
        x = self.conv44(x)

        x = self.conv51(x)
        x = self.conv52(x)
        x = self.conv53(x)
        x = self.conv54(x)
        # print("feature shape:", x.size())

        if self.include_top:
            x = x.view(x.size(0), -1)
            # TODO: why there is no dropout
            x = self.fc(x)
        return x


class VGG16(nn.Module):  # TODO: try different config of the channels
    def __init__(self, dropout_rate, num_classes, include_top, layer, type='none'):
        """
        :param type: this is only for shuffle experiments, remember to get rid of it afterwards!
        :param layer: int, if the conv number is smaller than the layer, normal conv is used; otherwise dconv
        """
        super(VGG16, self).__init__()
        print("CIFAR VGG16 is used")
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.include_top = include_top
        self.layer = layer

        # Define the building blocks
        if layer > 11:
            self.conv11 = CONV_3x3(3, 64, kernelsize=3, stride=1, padding='same', bias=False)
        else:
            self.conv11 = DCONV_3x3(3, 64, kernelsize=3, stride=1, padding=1, type=type)

        if layer > 12:
            self.conv12 = nn.Sequential(CONV_3x3(64, 64, kernelsize=3, stride=1, padding='same', bias=False),
                                        nn.MaxPool2d(kernel_size=2, stride=2))
            # self.conv12 = CONV_3x3(64, 64, kernelsize=3, stride=2, padding='same', bias=False)
        else:
            self.conv12 = DCONV_3x3(64, 64, kernelsize=3, stride=2, padding=1, type=type)

        if layer > 21:
            self.conv21 = CONV_3x3(64, 128, kernelsize=3, stride=1, padding='same', bias=False)
        else:
            self.conv21 = DCONV_3x3(64, 128, kernelsize=3, stride=1, padding=1, type=type)

        if layer > 22:
            self.conv22 = nn.Sequential(CONV_3x3(128, 128, kernelsize=3, stride=1, padding='same', bias=False),
                                        nn.MaxPool2d(kernel_size=2, stride=2))
            # self.conv22 = CONV_3x3(128, 128, kernelsize=3, stride=2, padding='same', bias=False)
        else:
            self.conv22 = DCONV_3x3(128, 128, kernelsize=3, stride=2, padding=1, type=type)

        if layer > 31:
            self.conv31 = CONV_3x3(128, 256, kernelsize=3, stride=1, padding='same', bias=False)
        else:
            self.conv31 = DCONV_3x3(128, 256, kernelsize=3, stride=1, padding=1, type=type)

        if layer > 32:
            self.conv32 = CONV_3x3(256, 256, kernelsize=3, stride=1, padding='same', bias=False)
        else:
            self.conv32 = DCONV_3x3(256, 256, kernelsize=3, stride=1, padding=1, type=type)

        if layer > 33:
            self.conv33 = nn.Sequential(CONV_3x3(256, 256, kernelsize=3, stride=1, padding='same', bias=False),
                                        nn.MaxPool2d(kernel_size=2, stride=2))
            # self.conv33 = CONV_3x3(256, 256, kernelsize=3, stride=2, padding='same', bias=False)
        else:
            self.conv33 = DCONV_3x3(256, 256, kernelsize=3, stride=2, padding=1, type=type)

        if layer > 41:
            self.conv41 = CONV_3x3(256, 512, kernelsize=3, stride=1, padding='same', bias=False)
        else:
            self.conv41 = DCONV_3x3(256, 512, kernelsize=3, stride=1, padding=1, type=type)

        if layer > 42:
            self.conv42 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding='same', bias=False)
        else:
            self.conv42 = DCONV_3x3(512, 512, kernelsize=3, stride=1, padding=1, type=type)

        if layer > 43:
            self.conv43 = nn.Sequential(CONV_3x3(512, 512, kernelsize=3, stride=1, padding='same', bias=False),
                                        nn.MaxPool2d(kernel_size=2, stride=2))
            # self.conv43 = CONV_3x3(512, 512, kernelsize=3, stride=2, padding='same', bias=False)
        else:
            self.conv43 = DCONV_3x3(512, 512, kernelsize=3, stride=2, padding=1, type=type)

        if layer > 51:
            self.conv51 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding='same', bias=False)
        else:
            self.conv51 = DCONV_3x3(512, 512, kernelsize=3, stride=1, padding=1, type=type)

        if layer > 52:
            self.conv52 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding='same', bias=False)
        else:
            self.conv52 = DCONV_3x3(512, 512, kernelsize=3, stride=1, padding=1, type=type)

        if layer > 53:
            self.conv53 = nn.Sequential(CONV_3x3(512, 512, kernelsize=3, stride=1, padding='same', bias=False),
                                        nn.MaxPool2d(kernel_size=2, stride=2))
            # self.conv53 = CONV_3x3(512, 512, kernelsize=3, stride=2, padding='same', bias=False)
        else:
            self.conv53 = DCONV_3x3(512, 512, kernelsize=3, stride=2, padding=1, type=type)

        # self.dropout = nn.Dropout(p=0.5)
        # self.fc = nn.Linear(512, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(512, 4096),
                                nn.ReLU(True),
                                nn.Linear(4096, 4096),
                                nn.ReLU(True),
                                nn.Linear(4096, num_classes))

        # Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                # raise Exception('You are using a model without BN!!!')
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_x):
        # print('input size:', input_x.size())
        x = self.conv11(input_x)
        x = self.conv12(x)

        x = self.conv21(x)
        x = self.conv22(x)

        x = self.conv31(x)
        x = self.conv32(x)
        x = self.conv33(x)

        x = self.conv41(x)
        x = self.conv42(x)
        x = self.conv43(x)

        x = self.conv51(x)
        x = self.conv52(x)
        x = self.conv53(x)
        x = self.avgpool(x)
        # print("feature shape:", x.size())

        if self.include_top:
            x = x.view(x.size(0), -1)
            # x = self.dropout(x)
            # TODO: why there is no dropout
            # np.save('/nethome/yuefan/fanyue/dconv/fcweight.npy', self.fc.weight.detach().cpu().numpy())
        #
        #     x_shuff = torch.empty(x.size(0), x.size(1)).cuda(0)
        #     perm = torch.randperm(x.size(1))
        #     x_shuff[:, :] = x[:, perm]
        #
        return self.fc(x)


class VGG16_dconv(nn.Module):  # TODO: try different config of the channels
    def __init__(self, dropout_rate, num_classes, include_top, layer, nblocks, type='none'):
        """
        only one layer can be shuffled here, 51, 52, 53 can't be shuffled
        :param type: this is only for shuffle experiments, remember to get rid of it afterwards!
        :param layer: int, if the conv number is smaller than the layer, normal conv is used; otherwise dconv
        """
        super(VGG16_dconv, self).__init__()
        print("CIFAR VGG16_dconv is used")
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.include_top = include_top
        self.layer = layer
        self.nrows = nblocks
        self.ncols = nblocks

        # Define the building blocks
        if layer == 11:
            self.conv11 = DCONV_3x3(3, 64, kernelsize=3, stride=1, padding=1, nrows=self.nrows, ncols=self.ncols,
                                    type=type)
        else:
            self.conv11 = CONV_3x3(3, 64, kernelsize=3, stride=1, padding='same', bias=False)

        if layer == 12:
            self.conv12 = DCONV_3x3(64, 64, kernelsize=3, stride=2, padding=1, nrows=self.nrows, ncols=self.ncols,
                                    type=type)
        else:
            self.conv12 = nn.Sequential(CONV_3x3(64, 64, kernelsize=3, stride=1, padding='same', bias=False),
                                        nn.MaxPool2d(kernel_size=2, stride=2))
            # self.conv12 = CONV_3x3(64, 64, kernelsize=3, stride=2, padding='same', bias=False)

        if layer == 21:
            self.conv21 = DCONV_3x3(64, 128, kernelsize=3, stride=1, padding=1, nrows=self.nrows, ncols=self.ncols,
                                    type=type)
        else:
            self.conv21 = CONV_3x3(64, 128, kernelsize=3, stride=1, padding='same', bias=False)

        if layer == 22:
            self.conv22 = DCONV_3x3(128, 128, kernelsize=3, stride=2, padding=1, nrows=self.nrows, ncols=self.ncols,
                                    type=type)
        else:
            self.conv22 = nn.Sequential(CONV_3x3(128, 128, kernelsize=3, stride=1, padding='same', bias=False),
                                        nn.MaxPool2d(kernel_size=2, stride=2))
            # self.conv22 = CONV_3x3(128, 128, kernelsize=3, stride=2, padding='same', bias=False)

        if layer == 31:
            self.conv31 = DCONV_3x3(128, 256, kernelsize=3, stride=1, padding=1, nrows=self.nrows, ncols=self.ncols,
                                    type=type)
        else:
            self.conv31 = CONV_3x3(128, 256, kernelsize=3, stride=1, padding='same', bias=False)

        if layer == 32:
            self.conv32 = DCONV_3x3(256, 256, kernelsize=3, stride=1, padding=1, nrows=self.nrows, ncols=self.ncols,
                                    type=type)
        else:
            self.conv32 = CONV_3x3(256, 256, kernelsize=3, stride=1, padding='same', bias=False)

        if layer == 33:
            self.conv33 = DCONV_3x3(256, 256, kernelsize=3, stride=2, padding=1, nrows=self.nrows, ncols=self.ncols,
                                    type=type)
        else:
            self.conv33 = nn.Sequential(CONV_3x3(256, 256, kernelsize=3, stride=1, padding='same', bias=False),
                                        nn.MaxPool2d(kernel_size=2, stride=2))
            # self.conv33 = CONV_3x3(256, 256, kernelsize=3, stride=2, padding='same', bias=False)

        if layer == 41:
            self.conv41 = DCONV_3x3(256, 512, kernelsize=3, stride=1, padding=1, nrows=self.nrows, ncols=self.ncols,
                                    type=type)
        else:
            self.conv41 = CONV_3x3(256, 512, kernelsize=3, stride=1, padding='same', bias=False)

        if layer == 42:
            self.conv42 = DCONV_3x3(512, 512, kernelsize=3, stride=1, padding=1, nrows=self.nrows, ncols=self.ncols,
                                    type=type)
        else:
            self.conv42 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding='same', bias=False)

        if layer == 43:
            self.conv43 = DCONV_3x3(512, 512, kernelsize=3, stride=2, padding=1, nrows=self.nrows, ncols=self.ncols,
                                    type=type)
        else:
            self.conv43 = nn.Sequential(CONV_3x3(512, 512, kernelsize=3, stride=1, padding='same', bias=False),
                                        nn.MaxPool2d(kernel_size=2, stride=2))
            # self.conv43 = CONV_3x3(512, 512, kernelsize=3, stride=2, padding='same', bias=False)

        if layer == 51:
            self.conv51 = DCONV_3x3(512, 512, kernelsize=3, stride=1, padding=1, nrows=self.nrows, ncols=self.ncols,
                                    type=type)
        else:
            self.conv51 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding='same', bias=False)

        if layer == 52:
            self.conv52 = DCONV_3x3(512, 512, kernelsize=3, stride=1, padding=1, nrows=self.nrows, ncols=self.ncols,
                                    type=type)
        else:
            self.conv52 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding='same', bias=False)

        if layer == 53:
            self.conv53 = DCONV_3x3(512, 512, kernelsize=3, stride=2, padding=1, nrows=self.nrows, ncols=self.ncols,
                                    type=type)
        else:
            self.conv53 = nn.Sequential(CONV_3x3(512, 512, kernelsize=3, stride=1, padding='same', bias=False),
                                        nn.MaxPool2d(kernel_size=2, stride=2))

        # self.dropout = nn.Dropout(p=0.5)
        # self.fc = nn.Linear(512, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(512, 4096),
                                nn.ReLU(True),
                                nn.Linear(4096, 4096),
                                nn.ReLU(True),
                                nn.Linear(4096, num_classes))

        # Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                # raise Exception('You are using a model without BN!!!')
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_x):
        # print('input size:', input_x.size())
        x = self.conv11(input_x)
        x = self.conv12(x)

        x = self.conv21(x)
        x = self.conv22(x)

        x = self.conv31(x)
        x = self.conv32(x)
        x = self.conv33(x)

        x = self.conv41(x)
        x = self.conv42(x)
        x = self.conv43(x)

        x = self.conv51(x)
        x = self.conv52(x)
        x = self.conv53(x)
        x = self.avgpool(x)
        # print("feature shape:", x.size())

        if self.include_top:
            x = x.view(x.size(0), -1)
            # x = self.dropout(x)
            # TODO: why there is no dropout
            # np.save('/nethome/yuefan/fanyue/dconv/fcweight.npy', self.fc.weight.detach().cpu().numpy())
        #
        #     x_shuff = torch.empty(x.size(0), x.size(1)).cuda(0)
        #     perm = torch.randperm(x.size(1))
        #     x_shuff[:, :] = x[:, perm]
        #
        return self.fc(x)


class VGG16_1d(nn.Module):
    def __init__(self, dropout_rate, num_classes, include_top, layer, is_shuff=False):
        """
        :param layer: int, if the conv number is smaller than the layer, normal conv is used; otherwise dconv
        :param is_shuff: boolean, whether using CONV1D_3x3 or CONV1Dshuff_3x3
        """
        super(VGG16_1d, self).__init__()
        print("CIFAR VGG16_1d is used")
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.include_top = include_top
        self.layer = layer
        self.bias = True
        self.ex = 1

        if is_shuff:
            conv_block = CONV1Dshuff_3x3
            print('shuff')
        else:
            conv_block = CONV1D_3x3

        # Define the building blocks
        if layer > 11:
            self.conv11 = CONV_3x3(3, 64*self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias)
        else:
            self.conv11 = conv_block(3, 64*self.ex, bias=self.bias)

        if layer > 12:
            # self.conv12 = CONV_3x3(64, 64, kernelsize=3, stride=2, padding='same', bias=False)
            self.conv12 = nn.Sequential(CONV_3x3(64*self.ex, 64*self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias),
                                        nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            self.conv12 = conv_block(64*self.ex, 64*self.ex, bias=self.bias)

        if layer > 21:
            self.conv21 = CONV_3x3(64*self.ex, 128*self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias)
        else:
            self.conv21 = conv_block(64*self.ex, 128*self.ex, bias=self.bias)

        if layer > 22:
            # self.conv22 = CONV_3x3(128, 128, kernelsize=3, stride=2, padding='same', bias=False)
            self.conv22 = nn.Sequential(CONV_3x3(128*self.ex, 128*self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias),
                                        nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            self.conv22 = conv_block(128*self.ex, 128*self.ex, bias=self.bias)

        if layer > 31:
            self.conv31 = CONV_3x3(128*self.ex, 256*self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias)
        else:
            self.conv31 = conv_block(128*self.ex, 256*self.ex, bias=self.bias)

        if layer > 32:
            self.conv32 = CONV_3x3(256*self.ex, 256*self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias)
        else:
            self.conv32 = conv_block(256*self.ex, 256*self.ex, bias=self.bias)

        if layer > 33:
            # self.conv33 = CONV_3x3(256, 256, kernelsize=3, stride=2, padding='same', bias=False)
            self.conv33 = nn.Sequential(CONV_3x3(256*self.ex, 256*self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias),
                                        nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            self.conv33 = conv_block(256*self.ex, 256*self.ex, bias=self.bias)

        if layer > 41:
            self.conv41 = CONV_3x3(256*self.ex, 512*self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias)
        else:
            self.conv41 = conv_block(256*self.ex, 512*self.ex, bias=self.bias)

        if layer > 42:
            self.conv42 = CONV_3x3(512*self.ex, 512*self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias)
        else:
            self.conv42 = conv_block(512*self.ex, 512*self.ex, bias=self.bias)

        if layer > 43:
            # self.conv43 = CONV_3x3(512, 512, kernelsize=3, stride=2, padding='same', bias=False)
            self.conv43 = nn.Sequential(CONV_3x3(512*self.ex, 512*self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias),
                                        nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            self.conv43 = conv_block(512*self.ex, 512*self.ex, bias=self.bias)

        if layer > 51:
            self.conv51 = CONV_3x3(512*self.ex, 512*self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias)
        else:
            self.conv51 = conv_block(512*self.ex, 512*self.ex, bias=self.bias)

        if layer > 52:
            self.conv52 = CONV_3x3(512*self.ex, 512*self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias)
        else:
            self.conv52 = conv_block(512*self.ex, 512*self.ex, bias=self.bias)

        if layer > 53:
            # self.conv53 = CONV_3x3(512, 512, kernelsize=3, stride=2, padding='same', bias=False)
            self.conv53 = nn.Sequential(CONV_3x3(512*self.ex, 512*self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias),
                                        nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            self.conv53 = conv_block(512*self.ex, 512*self.ex, bias=self.bias)

        # self.dropout = nn.Dropout(p=0.5)
        # if layer == 11 or layer == 12:
        #     s = 64
        # elif layer == 21 or layer == 22:
        #     s = 32
        # elif layer == 31 or layer == 32 or layer == 33:
        #     s = 16
        # elif layer == 41 or layer == 42 or layer == 43:
        #     s = 8
        # elif layer == 51 or layer == 52 or layer == 53:
        #     s = 4
        # else:
        #     s = 2
        # self.avgpool = nn.AvgPool2d(s)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Linear(512, num_classes)
        self.fc = nn.Sequential(nn.Linear(512*self.ex, 4096),
                                nn.ReLU(True),
                                nn.Linear(4096, 4096),
                                nn.ReLU(True),
                                nn.Linear(4096, num_classes))

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
        # print('input size:', input_x.size())
        if self.layer == 11:
            feat = x
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        x = self.conv11(x)
        if self.layer == 12:
            feat = x
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        x = self.conv12(x)
        if self.layer == 21:
            feat = x
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        x = self.conv21(x)
        if self.layer == 22:
            feat = x
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        x = self.conv22(x)
        if self.layer == 31:
            feat = x
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        x = self.conv31(x)
        if self.layer == 32:
            feat = x
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        x = self.conv32(x)
        if self.layer == 33:
            feat = x
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        x = self.conv33(x)
        if self.layer == 41:
            feat = x
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        x = self.conv41(x)
        if self.layer == 42:
            feat = x
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        x = self.conv42(x)
        if self.layer == 43:
            feat = x
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        x = self.conv43(x)
        if self.layer == 51:
            feat = x
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        x = self.conv51(x)
        if self.layer == 52:
            feat = x
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        x = self.conv52(x)
        if self.layer == 53:
            feat = x
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        x = self.conv53(x)
        # print("feature shape:", x.size())
        if self.layer == 99:
            x = self.avgpool(x)

        if self.include_top:
            x = x.view(x.size(0), -1)
            # x = self.dropout(x)
            # TODO: why there is no dropout
            # np.save('/nethome/yuefan/fanyue/dconv/weight52.npy', self.fc.weight.detach().cpu().numpy())

        return self.fc(x)


class VGG16_1x1(nn.Module):
    def __init__(self, dropout_rate, num_classes, include_top, layer):
        super(VGG16_1x1, self).__init__()
        print("CIFAR VGG16_1x1 is used")
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.include_top = include_top
        self.layer = layer
        self.bias = True
        self.ex = 1

        # Define the building blocks
        if layer > 11:
            self.conv11 = CONV_3x3(3, 64*self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias)
        else:
            self.conv11 = CONV_1x1(3, 64*self.ex, stride=1, padding=0, bias=self.bias)

        if layer > 12:
            self.conv12 = nn.Sequential(CONV_3x3(64*self.ex, 64*self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias),
                                        nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            self.conv12 = nn.Sequential(CONV_1x1(64*self.ex, 64*self.ex, stride=1, padding=0, bias=self.bias),
                                        nn.MaxPool2d(kernel_size=2, stride=2))

        if layer > 21:
            self.conv21 = CONV_3x3(64*self.ex, 128*self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias)
        else:
            self.conv21 = CONV_1x1(64*self.ex, 128*self.ex, stride=1, padding=0, bias=self.bias)

        if layer > 22:
            self.conv22 = nn.Sequential(CONV_3x3(128*self.ex, 128*self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias),
                                        nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            self.conv22 = nn.Sequential(CONV_1x1(128*self.ex, 128*self.ex, stride=1, padding=0, bias=self.bias),
                                        nn.MaxPool2d(kernel_size=2, stride=2))

        if layer > 31:
            self.conv31 = CONV_3x3(128*self.ex, 256*self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias)
        else:
            self.conv31 = CONV_1x1(128*self.ex, 256*self.ex, stride=1, padding=0, bias=self.bias)

        if layer > 32:
            self.conv32 = CONV_3x3(256*self.ex, 256*self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias)
        else:
            self.conv32 = CONV_1x1(256*self.ex, 256*self.ex, stride=1, padding=0, bias=self.bias)

        if layer > 33:
            self.conv33 = nn.Sequential(CONV_3x3(256*self.ex, 256*self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias),
                                        nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            self.conv33 = nn.Sequential(CONV_1x1(256*self.ex, 256*self.ex, stride=1, padding=0, bias=self.bias),
                                        nn.MaxPool2d(kernel_size=2, stride=2))

        if layer > 41:
            self.conv41 = CONV_3x3(256*self.ex, 512*self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias)
        else:
            self.conv41 = CONV_1x1(256*self.ex, 512*self.ex, stride=1, padding=0, bias=self.bias)

        if layer > 42:
            self.conv42 = CONV_3x3(512*self.ex, 512*self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias)
        else:
            self.conv42 = CONV_1x1(512*self.ex, 512*self.ex, stride=1, padding=0, bias=self.bias)

        if layer > 43:
            self.conv43 = nn.Sequential(CONV_3x3(512*self.ex, 512*self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias),
                                        nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            self.conv43 = nn.Sequential(CONV_1x1(512*self.ex, 512*self.ex, stride=1, padding=0, bias=self.bias),
                                        nn.MaxPool2d(kernel_size=2, stride=2))

        if layer > 51:
            self.conv51 = CONV_3x3(512*self.ex, 512*self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias)
        else:
            self.conv51 = CONV_1x1(512*self.ex, 512*self.ex, stride=1, padding=0, bias=self.bias)

        if layer > 52:
            self.conv52 = CONV_3x3(512*self.ex, 512*self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias)
        else:
            self.conv52 = CONV_1x1(512*self.ex, 512*self.ex, stride=1, padding=0, bias=self.bias)

        if layer > 53:
            self.conv53 = nn.Sequential(CONV_3x3(512*self.ex, 512*self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias),
                                        nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            self.conv53 = nn.Sequential(CONV_1x1(512*self.ex, 512*self.ex, stride=1, padding=0, bias=self.bias),
                                        nn.MaxPool2d(kernel_size=2, stride=2))

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(512*self.ex, 4096),
                                nn.ReLU(True),
                                nn.Linear(4096, 4096),
                                nn.ReLU(True),
                                nn.Linear(4096, num_classes))

        # Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                # raise Exception('You are using a model without BN!!!')
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                raise Exception('You are using a model without BN!!!')

    def forward(self, x):
        # print('input size:', input_x.size())
        x = self.conv11(x)
        x = self.conv12(x)

        x = self.conv21(x)
        x = self.conv22(x)

        x = self.conv31(x)
        x = self.conv32(x)
        x = self.conv33(x)

        x = self.conv41(x)
        x = self.conv42(x)
        x = self.conv43(x)

        x = self.conv51(x)
        x = self.conv52(x)
        x = self.conv53(x)

        if self.include_top:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        return x


class VGG16_truncated(nn.Module):
    # TODO: layer can not be 11 or 99!!!!!!!!!!!!!!!
    def __init__(self, dropout_rate, num_classes, include_top, layer):
        super(VGG16_truncated, self).__init__()
        print("CIFAR VGG16_truncated is used")
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.include_top = include_top
        self.layer = layer
        self.bias = True
        self.ex = 1
        self.modulelist = nn.ModuleList()

        # Define the building blocks
        if layer > 11:
            self.modulelist.append(CONV_3x3(3, 64 * self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias))
        if layer > 12:
            self.modulelist.append(nn.Sequential(
                CONV_3x3(64 * self.ex, 64 * self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias),
                nn.MaxPool2d(kernel_size=2, stride=2)))
        if layer > 21:
            self.modulelist.append(
                CONV_3x3(64 * self.ex, 128 * self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias))
        if layer > 22:
            self.modulelist.append(nn.Sequential(
                CONV_3x3(128 * self.ex, 128 * self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias),
                nn.MaxPool2d(kernel_size=2, stride=2)))
        if layer > 31:
            self.modulelist.append(
                CONV_3x3(128 * self.ex, 256 * self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias))
        if layer > 32:
            self.modulelist.append(
                CONV_3x3(256 * self.ex, 256 * self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias))
        if layer > 33:
            self.modulelist.append(nn.Sequential(
                CONV_3x3(256 * self.ex, 256 * self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias),
                nn.MaxPool2d(kernel_size=2, stride=2)))

        if layer > 41:
            self.modulelist.append(
                CONV_3x3(256 * self.ex, 512 * self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias))

        if layer > 42:
            self.modulelist.append(
                CONV_3x3(512 * self.ex, 512 * self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias))

        if layer > 43:
            self.modulelist.append(nn.Sequential(
                CONV_3x3(512 * self.ex, 512 * self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias),
                nn.MaxPool2d(kernel_size=2, stride=2)))

        if layer > 51:
            self.modulelist.append(
                CONV_3x3(512 * self.ex, 512 * self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias))

        if layer > 52:
            self.modulelist.append(
                CONV_3x3(512 * self.ex, 512 * self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias))

        if layer > 53:
            self.modulelist.append(nn.Sequential(
                CONV_3x3(512 * self.ex, 512 * self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias),
                nn.MaxPool2d(kernel_size=2, stride=2)))

        if layer == 42 or layer == 43 or layer == 51 or layer == 52 or layer == 53:
            s = 512 * self.ex
        elif layer == 32 or layer == 33 or layer == 41:
            s = 256 * self.ex
        elif layer == 31 or layer == 22:
            s = 128 * self.ex
        elif layer == 21 or layer == 11 or layer == 12:
            s = 64 * self.ex
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(s, 4096),
                                nn.ReLU(True),
                                nn.Linear(4096, 4096),
                                nn.ReLU(True),
                                nn.Linear(4096, num_classes))
        print(len(self.modulelist))

        # Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                # raise Exception('You are using a model without BN!!!')
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for i, module in enumerate(self.modulelist):
            # print("module %d has input feature shape:" % i, x.size())
            x = module(x)

        if self.include_top:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        return x


class VGG16_SA(nn.Module):  # TODO: try different config of the channels
    def __init__(self, dropout_rate, num_classes, include_top, layer, type='none'):
        """
        :param type: this is only for shuffle experiments, remember to get rid of it afterwards!
        :param layer: int, if the conv number is smaller than the layer, normal conv is used; otherwise dconv
        """
        super(VGG16_SA, self).__init__()
        print("CIFAR VGG16_SA is used")
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.include_top = include_top
        self.layer = layer

        # Define the building blocks
        self.conv11 = CONV_3x3(3, 64, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv12 = CONV_3x3(64, 64, kernelsize=3, stride=2, padding='same', bias=False)

        self.conv21 = CONV_3x3(64, 128, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv22 = CONV_3x3(128, 128, kernelsize=3, stride=2, padding='same', bias=False)

        self.conv31 = CONV_3x3(128, 256, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv32 = CONV_3x3(256, 256, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv33 = CONV_3x3(256, 256, kernelsize=3, stride=2, padding='same', bias=False)

        self.conv41 = CONV_3x3(256, 512, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv42 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv43 = CONV_3x3(512, 512, kernelsize=3, stride=2, padding='same', bias=False)

        self.conv51 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv52 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv53 = CONV_3x3(512, 512, kernelsize=3, stride=2, padding='same', bias=False)

        # self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(512, num_classes)

        # Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # print('input size:', input_x.size())
        if self.layer == 11:
            x = x * SpatialAttn_whr(x)
        x = self.conv11(x)
        if self.layer == 12:
            x = x * SpatialAttn_whr(x)
        x = self.conv12(x)
        if self.layer == 21:
            x = x * SpatialAttn_whr(x)
        x = self.conv21(x)
        if self.layer == 22:
            x = x * SpatialAttn_whr(x)
        x = self.conv22(x)
        if self.layer == 31:
            x = x * SpatialAttn_whr(x)
        x = self.conv31(x)
        if self.layer == 32:
            x = x * SpatialAttn_whr(x)
        x = self.conv32(x)
        if self.layer == 33:
            x = x * SpatialAttn_whr(x)
        x = self.conv33(x)
        if self.layer == 41:
            x = x * SpatialAttn_whr(x)
        x = self.conv41(x)
        if self.layer == 42:
            x = x * SpatialAttn_whr(x)
        x = self.conv42(x)
        if self.layer == 43:
            x = x * SpatialAttn_whr(x)
        x = self.conv43(x)
        if self.layer == 51:
            x = x * SpatialAttn_whr(x)
        x = self.conv51(x)
        if self.layer == 52:
            x = x * SpatialAttn_whr(x)
        x = self.conv52(x)
        if self.layer == 53:
            x = x * SpatialAttn_whr(x)
        x = self.conv53(x)
        # print("feature shape:", x.size())

        if self.include_top:
            x = x.view(x.size(0), -1)
            # x = self.dropout(x)
            # TODO: why there is no dropout
            # np.save('/nethome/yuefan/fanyue/dconv/fcweight.npy', self.fc.weight.detach().cpu().numpy())
        return self.fc(x)


class VGG16_3d(nn.Module):  # TODO: try different config of the channels
    def __init__(self, dropout_rate, num_classes, include_top):
        super(VGG16_3d, self).__init__()
        print("CIFAR VGG16_3d is used")
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.include_top = include_top

        # Define the building blocks
        self.conv11 = CONV_3x3(3, 64, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv12 = CONV_3x3(64, 64, kernelsize=3, stride=2, padding='same', bias=False)

        self.conv21 = CONV3D_3x3(64, 128, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv22 = CONV3D_3x3(128, 128, kernelsize=3, stride=2, padding='same', bias=False)

        self.conv31 = CONV_3x3(128, 256, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv32 = CONV_3x3(256, 256, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv33 = CONV_3x3(256, 256, kernelsize=3, stride=2, padding='same', bias=False)

        self.conv41 = CONV_3x3(256, 512, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv42 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv43 = CONV_3x3(512, 512, kernelsize=3, stride=2, padding='same', bias=False)

        self.conv51 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv52 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv53 = CONV_3x3(512, 512, kernelsize=3, stride=2, padding='same', bias=False)

        # self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(512, num_classes)

        # Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_x):
        # print('input size:', input_x.size())
        x = self.conv11(input_x)
        x = self.conv12(x)

        x = self.conv21(x)
        x = self.conv22(x)

        x = self.conv31(x)
        x = self.conv32(x)
        x = self.conv33(x)

        x = self.conv41(x)
        x = self.conv42(x)
        x = self.conv43(x)

        x = self.conv51(x)
        x = self.conv52(x)
        x = self.conv53(x)
        # print("feature shape:", x.size())

        if self.include_top:
            x = x.view(x.size(0), -1)
            # x = self.dropout(x)
            # TODO: why there is no dropout
            # np.save('/nethome/yuefan/fanyue/dconv/weight52.npy', self.fc.weight.detach().cpu().numpy())
            x = self.fc(x)
        return x


class VGG16_Transpose(nn.Module):  # TODO: try different config of the channels
    def __init__(self, dropout_rate, num_classes, include_top):
        super(VGG16_Transpose, self).__init__()
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.include_top = include_top

        # Define the building blocks
        self.conv11 = CONV_3x3(3, 64, kernelsize=3, stride=1, padding='same', bias=False)
        # self.conv11 = DCONV_3x3(3, 64)
        self.conv12 = CONV_3x3(64, 64, kernelsize=3, stride=2, padding='same', bias=False)

        self.conv21 = CONV_3x3(64, 128, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv22 = CONV_3x3(128, 128, kernelsize=3, stride=2, padding='same', bias=False)

        self.conv31 = CONV_3x3(128, 256, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv32 = CONV_3x3(256, 484, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv33 = CONV_3x3(64, 484, kernelsize=3, stride=2, padding='same', bias=False)

        self.conv41 = CONV_3x3(121, 484, kernelsize=3, stride=2, padding='same', bias=False)
        self.conv42 = CONV_3x3(121, 484, kernelsize=3, stride=2, padding='same', bias=False)
        # self.conv42 = DCONV_3x3(512, 512)
        self.conv43 = CONV_3x3(121, 484, kernelsize=3, stride=2, padding='same', bias=False)

        self.conv51 = CONV_3x3(121, 484, kernelsize=3, stride=2, padding='same', bias=False)
        self.conv52 = CONV_3x3(121, 484, kernelsize=3, stride=2, padding='same', bias=False)
        self.conv53 = CONV_3x3(121, 484, kernelsize=3, stride=2, padding='same', bias=False)

        # self.dropout = nn.Dropout(p=0.5)
        self.avgpool = nn.AvgPool2d(11)  # TODO: check the final size
        self.fc = nn.Linear(484, num_classes)

        # Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_x):
        # print('input size:', input_x.size())
        x = self.conv11(input_x)
        x = self.conv12(x)

        x = self.conv21(x)
        x = self.conv22(x)

        x = self.conv31(x)
        x = self.conv32(x)
        x = x.view(x.size(0), 22, 22, 64)
        x = x.permute(0, 3, 1, 2)
        x = self.conv33(x)

        x = x.view(x.size(0), 22, 22, 121)
        x = x.permute(0, 3, 1, 2)
        x = self.conv41(x)
        x = x.view(x.size(0), 22, 22, 121)
        x = x.permute(0, 3, 1, 2)
        x = self.conv42(x)
        x = x.view(x.size(0), 22, 22, 121)
        x = x.permute(0, 3, 1, 2)
        x = self.conv43(x)

        x = x.view(x.size(0), 22, 22, 121)
        x = x.permute(0, 3, 1, 2)
        x = self.conv51(x)
        x = x.view(x.size(0), 22, 22, 121)
        x = x.permute(0, 3, 1, 2)
        x = self.conv52(x)
        x = x.view(x.size(0), 22, 22, 121)
        x = x.permute(0, 3, 1, 2)
        x = self.conv53(x)
        # print("feature shape:", x.size())

        if self.include_top:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            # x = self.dropout(x)
            # TODO: why there is no dropout
            x = self.fc(x)
        return x


class VGG16_WHR(nn.Module):  # TODO: try different config of the channels
    def __init__(self, dropout_rate, num_classes, include_top, type='none'):
        """
        This is VGG16 PCB version
        :param type: this is only for shuffle experiments, remember to get rid of it afterwards!
        :param layer: int, if the conv number is smaller than the layer, normal conv is used; otherwise dconv
        """
        super(VGG16_WHR, self).__init__()
        print("CIFAR VGG16_WHR is used")
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.include_top = include_top
        self.num_features = 512

        # Define the building blocks
        self.conv11 = CONV_3x3(3, 64, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv12 = CONV_3x3(64, 64, kernelsize=3, stride=2, padding='same', bias=False)

        self.conv21 = CONV_3x3(64, 128, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv22 = CONV_3x3(128, 128, kernelsize=3, stride=2, padding='same', bias=False)

        self.conv31 = CONV_3x3(128, 256, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv32 = CONV_3x3(256, 256, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv33 = CONV_3x3(256, 256, kernelsize=3, stride=2, padding='same', bias=False)

        self.conv41 = CONV_3x3(256, 512, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv42 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv43 = CONV_3x3(512, 512, kernelsize=3, stride=2, padding='same', bias=False)

        self.conv51 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv52 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv53 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding='same', bias=False)
        # =======================================top=============================================
        self.instance0 = nn.Linear(self.num_features, self.num_classes)
        self.instance1 = nn.Linear(self.num_features, self.num_classes)
        self.instance2 = nn.Linear(self.num_features, self.num_classes)
        self.instance3 = nn.Linear(self.num_features, self.num_classes)
        self.instance4 = nn.Linear(self.num_features, self.num_classes)
        # self.linear_list = []
        # for i in range(16):
        #     self.linear_list.append(nn.Linear(self.num_features, self.num_classes).cuda())

        # Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = self.conv11(x)
        x = self.conv12(x)

        x = self.conv21(x)
        x = self.conv22(x)

        x = self.conv31(x)
        x = self.conv32(x)
        x = self.conv33(x)

        x = self.conv41(x)
        x = self.conv42(x)
        x = self.conv43(x)

        x = self.conv51(x)
        x = self.conv52(x)
        x = self.conv53(x)

        sx = x.size(2) / 2
        x = nn.functional.avg_pool2d(x, kernel_size=(sx, x.size(3)), stride=(sx, x.size(3)))  # 4x1

        x4 = nn.functional.avg_pool2d(x, kernel_size=(2, 1), stride=(1, 1))
        x4 = x4.contiguous().view(x4.size(0), -1)
        c4 = self.instance4(x4)

        # x = x.view(x.size(0), x.size(1), 16)
        # c_list = []
        # for i in range(16):
        #     x_offset = torch.empty(x.size(0), 512).cuda(0)
        #     # print(x_offset[:, :, :].size(), x[:, :, i].size())
        #     x_offset[:, :] = x[:, :, i]
        #     tmp = self.linear_list[i](x_offset)
        #     c_list.append(tmp)

        x = x.chunk(2, dim=2)
        x0 = x[0].contiguous().view(x[0].size(0), -1)
        x1 = x[1].contiguous().view(x[1].size(0), -1)
        c0 = self.instance0(x0)
        c1 = self.instance1(x1)
        return c0, c1, c4#c_list, c4#


def vggwhr(**kwargs):
    """
    Constructs a ResnetWHR model.
    """
    return VGG16_WHR(**kwargs)


def vgg19(**kwargs):
    """
    Constructs a VGG19 model.
    """
    return VGG19(**kwargs)


def vgg16(**kwargs):
    """
    Constructs a VGG16 model.
    """
    return VGG16(**kwargs)


def vgg16_3d(**kwargs):
    """
    Constructs a VGG16_3d model.
    """
    return VGG16_3d(**kwargs)


def vgg16_1d(**kwargs):
    """
    Constructs a VGG16_1d model.
    """
    return VGG16_1d(**kwargs)


def vgg16_1x1(**kwargs):
    """
    Constructs a VGG16_1x1 model.
    """
    return VGG16_1x1(**kwargs)


def vgg16_transpose(**kwargs):
    """
    Constructs a vgg16_transpose model.
    """
    return VGG16_Transpose(**kwargs)


def vgg16_sa(**kwargs):
    """
    Constructs a vgg16_sa model.
    """
    return VGG16_SA(**kwargs)


def vgg16_dconv(**kwargs):
    """
    Constructs a vgg16_sa model.
    """
    return VGG16_dconv(**kwargs)


def vgg16_truncated(**kwargs):
    """
    Constructs a VGG16_truncated model.
    """
    return VGG16_truncated(**kwargs)

# class VGG16_Transpose(nn.Module):  # TODO: try different config of the channels
#     def __init__(self, dropout_rate, num_classes, include_top):
#         super(VGG16_Transpose, self).__init__()
#         self.dropout_rate = dropout_rate
#         self.num_classes = num_classes
#         self.include_top = include_top
#
#         # Define the building blocks
#         self.conv11 = CONV_3x3(3, 64, kernelsize=3, stride=1, padding='same', bias=False)
#         self.conv12 = CONV_3x3(64, 64, kernelsize=3, stride=2, padding='same', bias=False)
#
#         self.conv21 = CONV_3x3(64, 128, kernelsize=3, stride=1, padding='same', bias=False)
#         self.conv22 = CONV_3x3(128, 128, kernelsize=3, stride=2, padding='same', bias=False)
#
#         self.conv31 = CONV_3x3(128, 256, kernelsize=3, stride=1, padding='same', bias=False)
#         self.conv32 = CONV_3x3(256, 256, kernelsize=3, stride=1, padding='same', bias=False)
#         self.conv33 = CONV_3x3(256, 256, kernelsize=3, stride=2, padding='same', bias=False)
#
#         self.conv41 = CONV_3x3(256, 512, kernelsize=3, stride=1, padding='same', bias=False)
#         self.conv42 = CONV_3x3(512, 484, kernelsize=3, stride=1, padding='same', bias=False)
#         # self.conv42 = DCONV_3x3(512, 512)
#         self.conv43 = CONV_3x3(16, 512, kernelsize=3, stride=2, padding='same', bias=False)
#
#         self.conv51 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding='same', bias=False)
#         self.conv52 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding='same', bias=False)
#         self.conv53 = CONV_3x3(512, 512, kernelsize=3, stride=2, padding='same', bias=False)
#
#         # self.dropout = nn.Dropout(p=0.5)
#         self.avgpool = nn.AvgPool2d(6)  # TODO: check the final size
#         self.fc = nn.Linear(512, num_classes)
#
#         # Initialize the weights
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#     def forward(self, input_x):
#         # print('input size:', input_x.size())
#         x = self.conv11(input_x)
#         x = self.conv12(x)
#
#         x = self.conv21(x)
#         x = self.conv22(x)
#
#         x = self.conv31(x)
#         x = self.conv32(x)
#         x = self.conv33(x)
#
#         x = self.conv41(x)
#
#         x = self.conv42(x)  # 128, 484, 4, 4
#         x = x.view(x.size(0), 22, 22, 16)
#         x = x.permute(0, 3, 1, 2)  # 128, 16, 22, 22
#         x = self.conv43(x)  # 128, 512, 11, 11
#
#         x = self.conv51(x)
#         x = self.conv52(x)
#         x = self.conv53(x)
#         # print("feature shape:", x.size())
#
#         if self.include_top:
#             x = self.avgpool(x)
#             x = x.view(x.size(0), -1)
#             # x = self.dropout(x)
#             # TODO: why there is no dropout
#             x = self.fc(x)
#         return x


# class VGG16_Transpose(nn.Module):  # TODO: try different config of the channels
#     # This one is the first one that achieves 52.44%!!!!
#     def __init__(self, dropout_rate, num_classes, include_top):
#         super(VGG16_Transpose, self).__init__()
#         self.dropout_rate = dropout_rate
#         self.num_classes = num_classes
#         self.include_top = include_top
#
#         # Define the building blocks
#         self.conv11 = CONV_3x3(3, 64, kernelsize=3, stride=1, padding='same', bias=False)
#         self.conv12 = CONV_3x3(64, 64, kernelsize=3, stride=2, padding='same', bias=False)
#
#         self.conv21 = CONV_3x3(64, 128, kernelsize=3, stride=1, padding='same', bias=False)
#         self.conv22 = CONV_3x3(128, 128, kernelsize=3, stride=2, padding='same', bias=False)
#
#         self.conv31 = CONV_3x3(128, 256, kernelsize=3, stride=1, padding='same', bias=False)
#         self.conv32 = CONV_3x3(256, 256, kernelsize=3, stride=1, padding='same', bias=False)
#         self.conv33 = CONV_3x3(256, 256, kernelsize=3, stride=2, padding='same', bias=False)
#
#         self.conv41 = CONV_3x3(256, 512, kernelsize=3, stride=1, padding='same', bias=False)
#         self.conv42 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding='same', bias=False)
#         # self.conv42 = DCONV_3x3(512, 512)
#         self.conv43 = CONV_3x3(512, 484, kernelsize=3, stride=2, padding='same', bias=False)
#
#         self.conv51 = CONV_3x3(4, 484, kernelsize=3, stride=2, padding='same', bias=False)
#         self.conv52 = CONV_3x3(121, 484, kernelsize=3, stride=2, padding='same', bias=False)
#         self.conv53 = CONV_3x3(121, 484, kernelsize=3, stride=2, padding='same', bias=False)
#
#         # self.dropout = nn.Dropout(p=0.5)
#         # self.avgpool = nn.AvgPool2d(6)  # TODO: check the final size
#         self.fc = nn.Linear(484, num_classes)
#
#         # Initialize the weights
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#     def forward(self, input_x):
#         # print('input size:', input_x.size())
#         x = self.conv11(input_x)
#         x = self.conv12(x)
#
#         x = self.conv21(x)
#         x = self.conv22(x)
#
#         x = self.conv31(x)
#         x = self.conv32(x)
#         x = self.conv33(x)
#
#         x = self.conv41(x)
#         x = self.conv42(x)
#         x = self.conv43(x)
#
#         x = x.view(x.size(0), 22, 22, 4)
#         x = x.permute(0, 3, 1, 2)  # [128, 4, 22, 22]
#         x = self.conv51(x)  # [128, 484, 11, 11]
#         x = x.view(x.size(0), 22, 22, 121)
#         x = x.permute(0, 3, 1, 2)
#         x = self.conv52(x)
#         x = x.view(x.size(0), 22, 22, 121)
#         x = x.permute(0, 3, 1, 2)  # [128, 121, 22, 22]
#         x = self.conv53(x)
#         x = x.view(x.size(0), 22, 22, 121)
#         x = x.permute(0, 3, 1, 2)  # [128, 121, 22, 22]
#         x = torch.mean(x, dim=1)
#         # print("feature shape:", x.size())
#
#         if self.include_top:
#             # x = self.avgpool(x)
#             x = x.view(x.size(0), -1)
#             # x = self.dropout(x)
#             # TODO: why there is no dropout
#             x = self.fc(x)
#         return x
