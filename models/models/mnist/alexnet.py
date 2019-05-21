import torch.nn as nn

__all__ = ['alexnet']


class conv_relu_maxpool(nn.Module):
    """
    This is just a wraper for a conv_relu_maxpool
    """
    def __init__(self, inplanes, outplanes, kernel_size, stride, padding, bias):
        super(conv_relu_maxpool, self).__init__()
        self.conv = nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class conv_relu(nn.Module):
    """
    This is just a wraper for a conv_relu
    """
    def __init__(self, inplanes, outplanes, kernel_size, stride, padding, bias):
        super(conv_relu, self).__init__()
        self.conv = nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class conv_relu_1d(nn.Module):
    """
    This is just a wraper for a conv_relu_1d
    """
    def __init__(self, inplanes, outplanes, bias):
        super(conv_relu_1d, self).__init__()
        self.conv = nn.Linear(inplanes, outplanes, bias=bias)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class AlexNet(nn.Module):
    def __init__(self, num_classes, layer):
        super(AlexNet, self).__init__()
        print('alexnet with layer %d is used' % layer)
        self.layer = layer
        if layer > 1:
            self.layer1 = conv_relu(1, 64, kernel_size=11, stride=4, padding=5, bias=True)
        else:
            self.layer1 = conv_relu_1d(1, 64, bias=True)
        if layer > 2:
            self.layer2 = conv_relu(64, 192, kernel_size=5, stride=1, padding=2, bias=True)
        else:
            self.layer2 = conv_relu_1d(64, 192, bias=True)
        if layer > 3:
            self.layer3 = conv_relu(192, 384, kernel_size=3, stride=1, padding=1, bias=True)
        else:
            self.layer3 = conv_relu_1d(192, 384, bias=True)
        if layer > 4:
            self.layer4 = conv_relu(384, 256, kernel_size=3, stride=1, padding=1, bias=True)
        else:
            self.layer4 = conv_relu_1d(384, 256, bias=True)
        if layer > 5:
            self.layer5 = conv_relu(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        else:
            self.layer5 = conv_relu_1d(256, 256, bias=True)
        # self.features = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        #
        #     nn.Conv2d(64, 192, kernel_size=5, padding=2),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        #
        #     nn.Conv2d(192, 384, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #
        #     nn.Conv2d(384, 256, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        # )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(256, 4096),
            nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        if self.layer == 1:
            # print("feature shape:", x.size())
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        x = self.layer1(x)
        # print("feature shape:", x.size())
        if self.layer == 2:
            # print("feature shape:", x.size())
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        x = self.layer2(x)
        # print("feature shape:", x.size())
        if self.layer == 3:
            # print("feature shape:", x.size())
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        x = self.layer3(x)
        if self.layer == 4:
            # print("feature shape:", x.size())
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        x = self.layer4(x)
        if self.layer == 5:
            # print("feature shape:", x.size())
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        x = self.layer5(x)
        if self.layer == 9:
            # print("feature shape:", x.size())
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def alexnet(**kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    model = AlexNet(**kwargs)
    return model
