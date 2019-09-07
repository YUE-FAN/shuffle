"""
This script contains vgg16 that has lowest and highest layers as parameters.
Only the layers inbetween will be modified.
"""
import torch.nn as nn


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


class VGG16_1x1LMP(nn.Module):
    def __init__(self, dropout_rate, num_classes, include_top, llayer, hlayer):
        super(VGG16_1x1LMP, self).__init__()
        print("CIFAR VGG16_1x1LMP is used")
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.include_top = include_top
        self.bias = True
        self.ex = 1
        assert llayer <= hlayer, 'low layer has to be smaller or equal to high layer'

        # Define the building blocks
        if llayer <= 11 <= hlayer:
            self.conv11 = CONV_1x1(3, 64 * self.ex, stride=1, padding=0, bias=self.bias)
        else:
            self.conv11 = CONV_3x3(3, 64 * self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias)

        if llayer <= 12 <= hlayer:
            self.conv12 = nn.Sequential(CONV_1x1(64 * self.ex, 64 * self.ex, stride=1, padding=0, bias=self.bias),
                                        nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            self.conv12 = nn.Sequential(
                CONV_3x3(64 * self.ex, 64 * self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias),
                nn.MaxPool2d(kernel_size=2, stride=2))

        if llayer <= 21 <= hlayer:
            self.conv21 = CONV_1x1(64 * self.ex, 128 * self.ex, stride=1, padding=0, bias=self.bias)
        else:
            self.conv21 = CONV_3x3(64 * self.ex, 128 * self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias)
        if llayer <= 22 <= hlayer:
            self.conv22 = nn.Sequential(CONV_1x1(128 * self.ex, 128 * self.ex, stride=1, padding=0, bias=self.bias),
                                        nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            self.conv22 = nn.Sequential(
                CONV_3x3(128 * self.ex, 128 * self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias),
                nn.MaxPool2d(kernel_size=2, stride=2))

        if llayer <= 31 <= hlayer:
            self.conv31 = CONV_1x1(128 * self.ex, 256 * self.ex, stride=1, padding=0, bias=self.bias)
        else:
            self.conv31 = CONV_3x3(128 * self.ex, 256 * self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias)
        if llayer <= 32 <= hlayer:
            self.conv32 = CONV_1x1(256 * self.ex, 256 * self.ex, stride=1, padding=0, bias=self.bias)
        else:
            self.conv32 = CONV_3x3(256 * self.ex, 256 * self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias)
        if llayer <= 33 <= hlayer:
            self.conv33 = nn.Sequential(CONV_1x1(256 * self.ex, 256 * self.ex, stride=1, padding=0, bias=self.bias),
                                        nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            self.conv33 = nn.Sequential(
                CONV_3x3(256 * self.ex, 256 * self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias),
                nn.MaxPool2d(kernel_size=2, stride=2))

        if llayer <= 41 <= hlayer:
            self.conv41 = CONV_1x1(256 * self.ex, 512 * self.ex, stride=1, padding=0, bias=self.bias)
        else:
            self.conv41 = CONV_3x3(256 * self.ex, 512 * self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias)
        if llayer <= 42 <= hlayer:
            self.conv42 = CONV_1x1(512 * self.ex, 512 * self.ex, stride=1, padding=0, bias=self.bias)
        else:
            self.conv42 = CONV_3x3(512 * self.ex, 512 * self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias)
        if llayer <= 43 <= hlayer:
            self.conv43 = nn.Sequential(CONV_1x1(512 * self.ex, 512 * self.ex, stride=1, padding=0, bias=self.bias),
                                        nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            self.conv43 = nn.Sequential(
                CONV_3x3(512 * self.ex, 512 * self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias),
                nn.MaxPool2d(kernel_size=2, stride=2))

        if llayer <= 51 <= hlayer:
            self.conv51 = CONV_1x1(512 * self.ex, 512 * self.ex, stride=1, padding=0, bias=self.bias)
        else:
            self.conv51 = CONV_3x3(512 * self.ex, 512 * self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias)
        if llayer <= 52 <= hlayer:
            self.conv52 = CONV_1x1(512 * self.ex, 512 * self.ex, stride=1, padding=0, bias=self.bias)
        else:
            self.conv52 = CONV_3x3(512 * self.ex, 512 * self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias)
        if llayer <= 53 <= hlayer:
            self.conv53 = nn.Sequential(CONV_1x1(512 * self.ex, 512 * self.ex, stride=1, padding=0, bias=self.bias),
                                        nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            self.conv53 = nn.Sequential(
                CONV_3x3(512 * self.ex, 512 * self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias),
                nn.MaxPool2d(kernel_size=2, stride=2))

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(512 * self.ex, 4096),
                                nn.ReLU(True),
                                nn.Linear(4096, 4096),
                                nn.ReLU(True),
                                nn.Linear(4096, num_classes))

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


class VGG16_1x1(nn.Module):
    def __init__(self, dropout_rate, num_classes, include_top, llayer, hlayer):
        super(VGG16_1x1, self).__init__()
        print("CIFAR VGG16_1x1 is used")
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.include_top = include_top
        self.bias = True
        self.ex = 1
        assert llayer <= hlayer, 'low layer has to be smaller or equal to high layer'

        # Define the building blocks
        if llayer <= 11 <= hlayer:
            self.conv11 = CONV_1x1(3, 64 * self.ex, stride=1, padding=0, bias=self.bias)
        else:
            self.conv11 = CONV_3x3(3, 64 * self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias)

        if llayer <= 12 <= hlayer:
            self.conv12 = CONV_1x1(64 * self.ex, 64 * self.ex, stride=2, padding=0, bias=self.bias)
        else:
            self.conv12 = nn.Sequential(
                CONV_3x3(64 * self.ex, 64 * self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias),
                nn.MaxPool2d(kernel_size=2, stride=2))

        if llayer <= 21 <= hlayer:
            self.conv21 = CONV_1x1(64 * self.ex, 128 * self.ex, stride=1, padding=0, bias=self.bias)
        else:
            self.conv21 = CONV_3x3(64 * self.ex, 128 * self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias)
        if llayer <= 22 <= hlayer:
            self.conv22 = CONV_1x1(128 * self.ex, 128 * self.ex, stride=2, padding=0, bias=self.bias)
        else:
            self.conv22 = nn.Sequential(
                CONV_3x3(128 * self.ex, 128 * self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias),
                nn.MaxPool2d(kernel_size=2, stride=2))

        if llayer <= 31 <= hlayer:
            self.conv31 = CONV_1x1(128 * self.ex, 256 * self.ex, stride=1, padding=0, bias=self.bias)
        else:
            self.conv31 = CONV_3x3(128 * self.ex, 256 * self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias)
        if llayer <= 32 <= hlayer:
            self.conv32 = CONV_1x1(256 * self.ex, 256 * self.ex, stride=1, padding=0, bias=self.bias)
        else:
            self.conv32 = CONV_3x3(256 * self.ex, 256 * self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias)
        if llayer <= 33 <= hlayer:
            self.conv33 = CONV_1x1(256 * self.ex, 256 * self.ex, stride=2, padding=0, bias=self.bias)
        else:
            self.conv33 = nn.Sequential(
                CONV_3x3(256 * self.ex, 256 * self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias),
                nn.MaxPool2d(kernel_size=2, stride=2))

        if llayer <= 41 <= hlayer:
            self.conv41 = CONV_1x1(256 * self.ex, 512 * self.ex, stride=1, padding=0, bias=self.bias)
        else:
            self.conv41 = CONV_3x3(256 * self.ex, 512 * self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias)
        if llayer <= 42 <= hlayer:
            self.conv42 = CONV_1x1(512 * self.ex, 512 * self.ex, stride=1, padding=0, bias=self.bias)
        else:
            self.conv42 = CONV_3x3(512 * self.ex, 512 * self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias)
        if llayer <= 43 <= hlayer:
            self.conv43 = CONV_1x1(512 * self.ex, 512 * self.ex, stride=2, padding=0, bias=self.bias)
        else:
            self.conv43 = nn.Sequential(
                CONV_3x3(512 * self.ex, 512 * self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias),
                nn.MaxPool2d(kernel_size=2, stride=2))

        if llayer <= 51 <= hlayer:
            self.conv51 = CONV_1x1(512 * self.ex, 512 * self.ex, stride=1, padding=0, bias=self.bias)
        else:
            self.conv51 = CONV_3x3(512 * self.ex, 512 * self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias)
        if llayer <= 52 <= hlayer:
            self.conv52 = CONV_1x1(512 * self.ex, 512 * self.ex, stride=1, padding=0, bias=self.bias)
        else:
            self.conv52 = CONV_3x3(512 * self.ex, 512 * self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias)
        if llayer <= 53 <= hlayer:
            self.conv53 = CONV_1x1(512 * self.ex, 512 * self.ex, stride=2, padding=0, bias=self.bias)
        else:
            self.conv53 = nn.Sequential(
                CONV_3x3(512 * self.ex, 512 * self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias),
                nn.MaxPool2d(kernel_size=2, stride=2))

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(512 * self.ex, 4096),
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


class VGG16_1x1DENSE(nn.Module):
    def __init__(self, dropout_rate, num_classes, include_top, llayer, hlayer):
        super(VGG16_1x1DENSE, self).__init__()
        print("VGG16_1x1DENSE is used")
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.include_top = include_top
        self.bias = True
        self.ex = 1
        assert llayer <= hlayer, 'low layer has to be smaller or equal to high layer'

        # Define the building blocks
        if llayer <= 11 <= hlayer:
            self.conv11 = CONV_1x1(3, 64 * self.ex, stride=1, padding=0, bias=self.bias)
        else:
            self.conv11 = CONV_3x3(3, 64 * self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias)

        if llayer <= 12 <= hlayer:
            self.conv12 = CONV_1x1(64 * self.ex, 64 * self.ex, stride=1, padding=0, bias=self.bias)
        else:
            self.conv12 = nn.Sequential(
                CONV_3x3(64 * self.ex, 64 * self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias),
                nn.MaxPool2d(kernel_size=2, stride=2))

        if llayer <= 21 <= hlayer:
            self.conv21 = CONV_1x1(64 * self.ex, 128 * self.ex, stride=1, padding=0, bias=self.bias)
        else:
            self.conv21 = CONV_3x3(64 * self.ex, 128 * self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias)
        if llayer <= 22 <= hlayer:
            self.conv22 = CONV_1x1(128 * self.ex, 128 * self.ex, stride=1, padding=0, bias=self.bias)
        else:
            self.conv22 = nn.Sequential(
                CONV_3x3(128 * self.ex, 128 * self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias),
                nn.MaxPool2d(kernel_size=2, stride=2))

        if llayer <= 31 <= hlayer:
            self.conv31 = CONV_1x1(128 * self.ex, 256 * self.ex, stride=1, padding=0, bias=self.bias)
        else:
            self.conv31 = CONV_3x3(128 * self.ex, 256 * self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias)
        if llayer <= 32 <= hlayer:
            self.conv32 = CONV_1x1(256 * self.ex, 256 * self.ex, stride=1, padding=0, bias=self.bias)
        else:
            self.conv32 = CONV_3x3(256 * self.ex, 256 * self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias)
        if llayer <= 33 <= hlayer:
            self.conv33 = CONV_1x1(256 * self.ex, 256 * self.ex, stride=1, padding=0, bias=self.bias)
        else:
            self.conv33 = nn.Sequential(
                CONV_3x3(256 * self.ex, 256 * self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias),
                nn.MaxPool2d(kernel_size=2, stride=2))

        if llayer <= 41 <= hlayer:
            self.conv41 = CONV_1x1(256 * self.ex, 512 * self.ex, stride=1, padding=0, bias=self.bias)
        else:
            self.conv41 = CONV_3x3(256 * self.ex, 512 * self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias)
        if llayer <= 42 <= hlayer:
            self.conv42 = CONV_1x1(512 * self.ex, 512 * self.ex, stride=1, padding=0, bias=self.bias)
        else:
            self.conv42 = CONV_3x3(512 * self.ex, 512 * self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias)
        if llayer <= 43 <= hlayer:
            self.conv43 = CONV_1x1(512 * self.ex, 512 * self.ex, stride=1, padding=0, bias=self.bias)
        else:
            self.conv43 = nn.Sequential(
                CONV_3x3(512 * self.ex, 512 * self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias),
                nn.MaxPool2d(kernel_size=2, stride=2))

        if llayer <= 51 <= hlayer:
            self.conv51 = CONV_1x1(512 * self.ex, 512 * self.ex, stride=1, padding=0, bias=self.bias)
        else:
            self.conv51 = CONV_3x3(512 * self.ex, 512 * self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias)
        if llayer <= 52 <= hlayer:
            self.conv52 = CONV_1x1(512 * self.ex, 512 * self.ex, stride=1, padding=0, bias=self.bias)
        else:
            self.conv52 = CONV_3x3(512 * self.ex, 512 * self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias)
        if llayer <= 53 <= hlayer:
            self.conv53 = CONV_1x1(512 * self.ex, 512 * self.ex, stride=1, padding=0, bias=self.bias)
        else:
            self.conv53 = nn.Sequential(
                CONV_3x3(512 * self.ex, 512 * self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias),
                nn.MaxPool2d(kernel_size=2, stride=2))

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(512 * self.ex, 4096),
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


class VGG16_1x1LAP(nn.Module):
    def __init__(self, dropout_rate, num_classes, include_top, llayer, hlayer):
        super(VGG16_1x1LAP, self).__init__()
        print("CIFAR VGG16_1x1LAP is used")
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.include_top = include_top
        self.bias = True
        self.ex = 1
        assert llayer <= hlayer, 'low layer has to be smaller or equal to high layer'

        # Define the building blocks
        if llayer <= 11 <= hlayer:
            self.conv11 = CONV_1x1(3, 64 * self.ex, stride=1, padding=0, bias=self.bias)
        else:
            self.conv11 = CONV_3x3(3, 64 * self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias)

        if llayer <= 12 <= hlayer:
            self.conv12 = nn.Sequential(CONV_1x1(64 * self.ex, 64 * self.ex, stride=1, padding=0, bias=self.bias),
                                        nn.AvgPool2d(kernel_size=2, stride=2))
        else:
            self.conv12 = nn.Sequential(
                CONV_3x3(64 * self.ex, 64 * self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias),
                nn.MaxPool2d(kernel_size=2, stride=2))

        if llayer <= 21 <= hlayer:
            self.conv21 = CONV_1x1(64 * self.ex, 128 * self.ex, stride=1, padding=0, bias=self.bias)
        else:
            self.conv21 = CONV_3x3(64 * self.ex, 128 * self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias)
        if llayer <= 22 <= hlayer:
            self.conv22 = nn.Sequential(CONV_1x1(128 * self.ex, 128 * self.ex, stride=1, padding=0, bias=self.bias),
                                        nn.AvgPool2d(kernel_size=2, stride=2))
        else:
            self.conv22 = nn.Sequential(
                CONV_3x3(128 * self.ex, 128 * self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias),
                nn.MaxPool2d(kernel_size=2, stride=2))

        if llayer <= 31 <= hlayer:
            self.conv31 = CONV_1x1(128 * self.ex, 256 * self.ex, stride=1, padding=0, bias=self.bias)
        else:
            self.conv31 = CONV_3x3(128 * self.ex, 256 * self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias)
        if llayer <= 32 <= hlayer:
            self.conv32 = CONV_1x1(256 * self.ex, 256 * self.ex, stride=1, padding=0, bias=self.bias)
        else:
            self.conv32 = CONV_3x3(256 * self.ex, 256 * self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias)
        if llayer <= 33 <= hlayer:
            self.conv33 = nn.Sequential(CONV_1x1(256 * self.ex, 256 * self.ex, stride=1, padding=0, bias=self.bias),
                                        nn.AvgPool2d(kernel_size=2, stride=2))
        else:
            self.conv33 = nn.Sequential(
                CONV_3x3(256 * self.ex, 256 * self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias),
                nn.MaxPool2d(kernel_size=2, stride=2))

        if llayer <= 41 <= hlayer:
            self.conv41 = CONV_1x1(256 * self.ex, 512 * self.ex, stride=1, padding=0, bias=self.bias)
        else:
            self.conv41 = CONV_3x3(256 * self.ex, 512 * self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias)
        if llayer <= 42 <= hlayer:
            self.conv42 = CONV_1x1(512 * self.ex, 512 * self.ex, stride=1, padding=0, bias=self.bias)
        else:
            self.conv42 = CONV_3x3(512 * self.ex, 512 * self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias)
        if llayer <= 43 <= hlayer:
            self.conv43 = nn.Sequential(CONV_1x1(512 * self.ex, 512 * self.ex, stride=1, padding=0, bias=self.bias),
                                        nn.AvgPool2d(kernel_size=2, stride=2))
        else:
            self.conv43 = nn.Sequential(
                CONV_3x3(512 * self.ex, 512 * self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias),
                nn.MaxPool2d(kernel_size=2, stride=2))

        if llayer <= 51 <= hlayer:
            self.conv51 = CONV_1x1(512 * self.ex, 512 * self.ex, stride=1, padding=0, bias=self.bias)
        else:
            self.conv51 = CONV_3x3(512 * self.ex, 512 * self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias)
        if llayer <= 52 <= hlayer:
            self.conv52 = CONV_1x1(512 * self.ex, 512 * self.ex, stride=1, padding=0, bias=self.bias)
        else:
            self.conv52 = CONV_3x3(512 * self.ex, 512 * self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias)
        if llayer <= 53 <= hlayer:
            self.conv53 = nn.Sequential(CONV_1x1(512 * self.ex, 512 * self.ex, stride=1, padding=0, bias=self.bias),
                                        nn.AvgPool2d(kernel_size=2, stride=2))
        else:
            self.conv53 = nn.Sequential(
                CONV_3x3(512 * self.ex, 512 * self.ex, kernelsize=3, stride=1, padding='same', bias=self.bias),
                nn.MaxPool2d(kernel_size=2, stride=2))

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(512 * self.ex, 4096),
                                nn.ReLU(True),
                                nn.Linear(4096, 4096),
                                nn.ReLU(True),
                                nn.Linear(4096, num_classes))

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


def vgg16_1x1dense(**kwargs):
    """
    Constructs a VGG16_1x1DENSE model.
    """
    return VGG16_1x1DENSE(**kwargs)


def vgg16_1x1lmp(**kwargs):
    """
    Constructs a VGG16_1x1LMP model.
    """
    return VGG16_1x1LMP(**kwargs)


def vgg16_1x1lap(**kwargs):
    """
    Constructs a VGG16_1x1LAP model.
    """
    return VGG16_1x1LAP(**kwargs)


def vgg16_1x1(**kwargs):
    """
    Constructs a VGG16_1x1 model.
    """
    return VGG16_1x1(**kwargs)
