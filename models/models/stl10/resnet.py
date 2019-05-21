import torch.nn as nn
from .dconv import DConv1Dai_share, DConv1Dai, Dconv_cos, Dconv_euc


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



def conv_1_3x3():
    return nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),  # 'SAME'
                         nn.BatchNorm2d(32),
                         nn.ReLU(inplace=True))
                         # TODO: nn.MaxPool2d(kernel_size=3, stride=2, padding=0))  # 'valid'


def conv_1_3x3_dconv():
    return nn.Sequential(Dconv_cos(32, 32),
                         # DConv1Dai(3),
                         # nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),  # 'SAME'
                         nn.BatchNorm2d(16),
                         nn.ReLU(inplace=True))
                         # TODO: nn.MaxPool2d(kernel_size=3, stride=2, padding=0))  # 'valid'


class identity_block3(nn.Module):
    def __init__(self, inplanes, planes, kernel_size):
        super(identity_block3, self).__init__()
        plane1, plane2, plane3 = planes
        self.conv1 = nn.Conv2d(inplanes, plane1, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(plane1)
        self.conv2 = nn.Conv2d(plane1, plane2, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) / 2, bias=False)
        self.bn2 = nn.BatchNorm2d(plane2)
        self.conv3 = nn.Conv2d(plane2, plane3, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(plane3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_tensor):
        out = self.conv1(input_tensor)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += input_tensor
        out = self.relu(out)
        return out


class identity_block3_dconv(nn.Module):
    def __init__(self, inplanes, planes, kernel_size):
        super(identity_block3_dconv, self).__init__()
        plane1, plane2, plane3 = planes
        self.conv1 = nn.Conv2d(inplanes, plane1, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(plane1)

        # self.offset = DConv1Dai(plane1)
        # self.conv2 = nn.Conv2d(plane1, plane2, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) / 2, bias=False)
        self.dconv = Dconv_cos(8, 8, plane1, plane2)  # TODO: don't use hard code here
        self.bn2 = nn.BatchNorm2d(plane2)

        self.conv3 = nn.Conv2d(plane2, plane3, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(plane3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_tensor):
        out = self.conv1(input_tensor)
        out = self.bn1(out)
        out = self.relu(out)

        # out = self.offset(out)
        # out = self.conv2(out)
        out = self.dconv(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += input_tensor
        out = self.relu(out)
        return out


class bottleneck(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, strides=(2, 2)):
        super(bottleneck, self).__init__()
        plane1, plane2, plane3 = planes
        self.conv1 = nn.Conv2d(inplanes, plane1, kernel_size=1, stride=strides, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(plane1)
        self.conv2 = nn.Conv2d(plane1, plane2, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) / 2, bias=False)
        self.bn2 = nn.BatchNorm2d(plane2)
        self.conv3 = nn.Conv2d(plane2, plane3, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(plane3)
        self.conv4 = nn.Conv2d(inplanes, plane3, kernel_size=1, stride=strides, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(plane3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_tensor):
        out = self.conv1(input_tensor)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        shortcut = self.conv4(input_tensor)
        shortcut = self.bn4(shortcut)

        out += shortcut
        out = self.relu(out)
        return out


class bottleneck_dconv(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, strides=(2, 2)):
        super(bottleneck_dconv, self).__init__()
        plane1, plane2, plane3 = planes
        self.conv1 = nn.Conv2d(inplanes, plane1, kernel_size=1, stride=strides, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(plane1)

        # self.offset = DConv1Dai(plane1)
        # self.conv2 = nn.Conv2d(plane1, plane2, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) / 2, bias=False)
        self.dconv = Dconv_cos(8, 8, plane1, plane2)  # TODO: don't use hard code here
        self.bn2 = nn.BatchNorm2d(plane2)

        self.conv3 = nn.Conv2d(plane2, plane3, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(plane3)

        self.conv4 = nn.Conv2d(inplanes, plane3, kernel_size=1, stride=strides, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(plane3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_tensor):
        out = self.conv1(input_tensor)
        out = self.bn1(out)
        out = self.relu(out)

        # out = self.offset(out)
        # out = self.conv2(out)
        out = self.dconv(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        shortcut = self.conv4(input_tensor)
        shortcut = self.bn4(shortcut)

        out += shortcut
        out = self.relu(out)
        return out

# def basic_block(input_tensor, kernel_size, filters, stage, block, training, reuse, strides=(2, 2)):
#     filters1, filters2 = filters
#     conv_name_base = 'res' + str(stage) + block + '_branch'
#     bn_name_base = 'bn' + str(stage) + block + '_branch'
#
#     x = tf.layers.conv2d(inputs=input_tensor, filters=filters1, kernel_size=kernel_size, strides=strides,
#                          padding='SAME', use_bias=False, reuse=reuse, name=conv_name_base + '2a')
#     x = tf.layers.batch_normalization(x, training=training, reuse=reuse, name=bn_name_base + '2a')
#     x = tf.nn.relu(x)
#
#     x = tf.layers.conv2d(inputs=x, filters=filters2, kernel_size=kernel_size, strides=1,
#                          padding='SAME', use_bias=False, reuse=reuse, name=conv_name_base + '2b')
#     x = tf.layers.batch_normalization(x, training=training, reuse=reuse, name=bn_name_base + '2b')
#
#     if strides == (1, 1):
#         shortcut = input_tensor
#     else:
#         shortcut = tf.layers.conv2d(inputs=input_tensor, filters=filters2, kernel_size=1, strides=strides,
#                                     padding='SAME', use_bias=False, reuse=reuse, name=conv_name_base + '1')
#         shortcut = tf.layers.batch_normalization(shortcut, training=training, reuse=reuse, name=bn_name_base + '1')
#
#     x = x + shortcut
#     x = tf.nn.relu(x)
#     return x
#
#
# def identity_block2(input_tensor, kernel_size, filters, stage, block, training, reuse):
#     filters1, filters2 = filters
#     conv_name_base = 'res' + str(stage) + block + '_branch'
#     bn_name_base = 'bn' + str(stage) + block + '_branch'
#
#     x = tf.layers.conv2d(inputs=input_tensor, filters=filters1, kernel_size=kernel_size, strides=1,
#                          padding='SAME', use_bias=False, reuse=reuse, name=conv_name_base + '2a')
#     x = tf.layers.batch_normalization(x, training=training, reuse=reuse, name=bn_name_base + '2a')
#     x = tf.nn.relu(x)
#
#     x = tf.layers.conv2d(inputs=x, filters=filters2, kernel_size=kernel_size, strides=1,
#                          padding='SAME', use_bias=False, reuse=reuse, name=conv_name_base + '2b')
#     x = tf.layers.batch_normalization(x, training=training, reuse=reuse, name=bn_name_base + '2b')
#
#     x = x + input_tensor
#     x = tf.nn.relu(x)
#
#     return x


class Resnet50(nn.Module):
    def __init__(self, dropout_rate, num_classes, include_top):
        super(Resnet50, self).__init__()
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.include_top = include_top

        # Define the building blocks
        self.conv_3x3 = conv_1_3x3()  # TODO: check if you are using dconv3x3

        self.bottleneck_1 = bottleneck(16, [16, 16, 64], kernel_size=3, strides=(1, 1))
        self.identity_block_1_1 = identity_block3(64, [16, 16, 64], kernel_size=3)
        self.identity_block_1_2 = identity_block3(64, [16, 16, 64], kernel_size=3)

        self.bottleneck_2 = bottleneck(64, [32, 32, 128], kernel_size=3, strides=(2, 2))
        self.identity_block_2_1 = identity_block3(128, [32, 32, 128], kernel_size=3)
        self.identity_block_2_2 = identity_block3(128, [32, 32, 128], kernel_size=3)
        self.identity_block_2_3 = identity_block3(128, [32, 32, 128], kernel_size=3)

        self.bottleneck_3 = bottleneck(128, [64, 64, 256], kernel_size=3, strides=(2, 2))
        self.identity_block_3_1 = identity_block3(256, [64, 64, 256], kernel_size=3)
        self.identity_block_3_2 = identity_block3(256, [64, 64, 256], kernel_size=3)
        self.identity_block_3_3 = identity_block3(256, [64, 64, 256], kernel_size=3)
        self.identity_block_3_4 = identity_block3(256, [64, 64, 256], kernel_size=3)
        self.identity_block_3_5 = identity_block3(256, [64, 64, 256], kernel_size=3)

        self.bottleneck_4 = bottleneck(256, [128, 128, 512], kernel_size=3, strides=(2, 2))
        self.identity_block_4_1 = identity_block3(512, [128, 128, 512], kernel_size=3)
        self.identity_block_4_2 = identity_block3(512, [128, 128, 512], kernel_size=3)

        self.avgpool = nn.AvgPool2d(6)  # TODO: check the final size
        self.fc = nn.Linear(512, num_classes)

        # Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_x):
        # print(input_x.size())
        x = self.conv_3x3(input_x)
        # print(x.size())
        x = self.bottleneck_1(x)
        x = self.identity_block_1_1(x)
        x = self.identity_block_1_2(x)
        # print(x.size())
        x = self.bottleneck_2(x)
        x = self.identity_block_2_1(x)
        x = self.identity_block_2_2(x)
        x = self.identity_block_2_3(x)
        # print(x.size())
        x = self.bottleneck_3(x)
        x = self.identity_block_3_1(x)
        x = self.identity_block_3_2(x)
        x = self.identity_block_3_3(x)
        x = self.identity_block_3_4(x)
        x = self.identity_block_3_5(x)
        # print(x.size())
        x = self.bottleneck_4(x)
        x = self.identity_block_4_1(x)
        x = self.identity_block_4_2(x)
        # print("feature shape:", x.size())

        if self.include_top:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            # TODO: why there is no dropout
            x = self.fc(x)
        return x


class DconvResnet50(nn.Module):
    def __init__(self, dropout_rate, num_classes, include_top):
        super(DconvResnet50, self).__init__()
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.include_top = include_top

        # Define the building blocks
        self.conv_3x3 = conv_1_3x3()

        self.bottleneck_1 = bottleneck(16, [16, 16, 64], kernel_size=3, strides=(1, 1))
        self.identity_block_1_1 = identity_block3(64, [16, 16, 64], kernel_size=3)
        self.identity_block_1_2 = identity_block3(64, [16, 16, 64], kernel_size=3)

        self.bottleneck_2 = bottleneck(64, [32, 32, 128], kernel_size=3, strides=(2, 2))
        self.identity_block_2_1 = identity_block3(128, [32, 32, 128], kernel_size=3)
        self.identity_block_2_2 = identity_block3(128, [32, 32, 128], kernel_size=3)
        self.identity_block_2_3 = identity_block3(128, [32, 32, 128], kernel_size=3)

        self.bottleneck_3 = bottleneck(128, [64, 64, 256], kernel_size=3, strides=(2, 2))
        self.identity_block_3_1 = identity_block3(256, [64, 64, 256], kernel_size=3)
        self.identity_block_3_2 = identity_block3(256, [64, 64, 256], kernel_size=3)
        self.identity_block_3_3 = identity_block3(256, [64, 64, 256], kernel_size=3)
        self.identity_block_3_4 = identity_block3(256, [64, 64, 256], kernel_size=3)
        self.identity_block_3_5 = identity_block3(256, [64, 64, 256], kernel_size=3)

        self.bottleneck_4 = bottleneck_dconv(256, [128, 128, 512], kernel_size=3, strides=(1, 1))
        self.identity_block_4_1 = identity_block3_dconv(512, [128, 128, 512], kernel_size=3)
        self.identity_block_4_2 = identity_block3_dconv(512, [128, 128, 512], kernel_size=3)

        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(512, num_classes)

        # Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_x):
        # print(input_x.size())
        x = self.conv_3x3(input_x)
        # print(x.size())
        x = self.bottleneck_1(x)
        x = self.identity_block_1_1(x)
        x = self.identity_block_1_2(x)
        # print(x.size())
        x = self.bottleneck_2(x)
        x = self.identity_block_2_1(x)
        x = self.identity_block_2_2(x)
        x = self.identity_block_2_3(x)
        # print(x.size())
        x = self.bottleneck_3(x)
        x = self.identity_block_3_1(x)
        x = self.identity_block_3_2(x)
        x = self.identity_block_3_3(x)
        x = self.identity_block_3_4(x)
        x = self.identity_block_3_5(x)
        # print(x.size())
        x = self.bottleneck_4(x)
        x = self.identity_block_4_1(x)
        x = self.identity_block_4_2(x)
        # print("feature shape:", x.size())

        if self.include_top:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            # TODO: why there is no dropout
            x = self.fc(x)
        return x


class Resnet50_SA(nn.Module):
    def __init__(self, dropout_rate, num_classes, include_top):
        super(Resnet50_SA, self).__init__()
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.include_top = include_top

        # Define the building blocks
        self.conv_3x3 = conv_1_3x3()

        self.bottleneck_1 = bottleneck(16, [16, 16, 64], kernel_size=3, strides=(1, 1))
        self.identity_block_1_1 = identity_block3(64, [16, 16, 64], kernel_size=3)
        self.identity_block_1_2 = identity_block3(64, [16, 16, 64], kernel_size=3)

        self.bottleneck_2 = bottleneck(64, [32, 32, 128], kernel_size=3, strides=(2, 2))
        self.identity_block_2_1 = identity_block3(128, [32, 32, 128], kernel_size=3)
        self.identity_block_2_2 = identity_block3(128, [32, 32, 128], kernel_size=3)
        self.identity_block_2_3 = identity_block3(128, [32, 32, 128], kernel_size=3)

        self.bottleneck_3 = bottleneck(128, [64, 64, 256], kernel_size=3, strides=(2, 2))
        self.identity_block_3_1 = identity_block3(256, [64, 64, 256], kernel_size=3)
        self.identity_block_3_2 = identity_block3(256, [64, 64, 256], kernel_size=3)
        self.identity_block_3_3 = identity_block3(256, [64, 64, 256], kernel_size=3)
        self.identity_block_3_4 = identity_block3(256, [64, 64, 256], kernel_size=3)
        self.identity_block_3_5 = identity_block3(256, [64, 64, 256], kernel_size=3)

        self.bottleneck_4 = bottleneck(256, [128, 128, 512], kernel_size=3, strides=(1, 1))
        self.identity_block_4_1 = identity_block3(512, [128, 128, 512], kernel_size=3)
        self.identity_block_4_2 = identity_block3(512, [128, 128, 512], kernel_size=3)

        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(512, num_classes)

        # Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_x):
        # print(input_x.size())
        x = self.conv_3x3(input_x)
        # print(x.size())
        x = self.bottleneck_1(x)
        x = self.identity_block_1_1(x)
        x = self.identity_block_1_2(x)
        # print(x.size())
        x = x * SpatialAttn_whr(x)
        x = self.bottleneck_2(x)
        x = self.identity_block_2_1(x)
        x = self.identity_block_2_2(x)
        x = self.identity_block_2_3(x)
        # print(x.size())
        x = self.bottleneck_3(x)
        x = self.identity_block_3_1(x)
        x = self.identity_block_3_2(x)
        x = self.identity_block_3_3(x)
        x = self.identity_block_3_4(x)
        x = self.identity_block_3_5(x)
        # print(x.size())

        x = self.bottleneck_4(x)
        x = self.identity_block_4_1(x)
        x = self.identity_block_4_2(x)
        # print("feature shape:", x.size())

        if self.include_top:
            # y = x[0, :, :, :]
            # y = y.sum(0)
            # z = torch.round(y)
            # print(z)
            # x = x * SpatialAttn_whr(x)
            # y = x[0, :, :, :]
            # y = y.sum(0)
            # z = torch.round(y)
            # print(z)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            # TODO: why there is no dropout
            x = self.fc(x)
        return x


class Resnet50_CA(nn.Module):
    def __init__(self, dropout_rate, num_classes, include_top):
        super(Resnet50_CA, self).__init__()
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.include_top = include_top

        # Define the building blocks
        self.conv_3x3 = conv_1_3x3()

        self.bottleneck_1 = bottleneck(16, [16, 16, 64], kernel_size=3, strides=(1, 1))
        self.identity_block_1_1 = identity_block3(64, [16, 16, 64], kernel_size=3)
        self.identity_block_1_2 = identity_block3(64, [16, 16, 64], kernel_size=3)

        self.bottleneck_2 = bottleneck(64, [32, 32, 128], kernel_size=3, strides=(2, 2))
        self.identity_block_2_1 = identity_block3(128, [32, 32, 128], kernel_size=3)
        self.identity_block_2_2 = identity_block3(128, [32, 32, 128], kernel_size=3)
        self.identity_block_2_3 = identity_block3(128, [32, 32, 128], kernel_size=3)

        self.bottleneck_3 = bottleneck(128, [64, 64, 256], kernel_size=3, strides=(2, 2))
        self.identity_block_3_1 = identity_block3(256, [64, 64, 256], kernel_size=3)
        self.identity_block_3_2 = identity_block3(256, [64, 64, 256], kernel_size=3)
        self.identity_block_3_3 = identity_block3(256, [64, 64, 256], kernel_size=3)
        self.identity_block_3_4 = identity_block3(256, [64, 64, 256], kernel_size=3)
        self.identity_block_3_5 = identity_block3(256, [64, 64, 256], kernel_size=3)

        self.bottleneck_4 = bottleneck(256, [128, 128, 512], kernel_size=3, strides=(1, 1))
        self.identity_block_4_1 = identity_block3(512, [128, 128, 512], kernel_size=3)
        self.identity_block_4_2 = identity_block3(512, [128, 128, 512], kernel_size=3)

        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(512, num_classes)

        # Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_x):
        # print(input_x.size())
        x = self.conv_3x3(input_x)
        # print(x.size())
        x = self.bottleneck_1(x)
        x = self.identity_block_1_1(x)
        x = self.identity_block_1_2(x)
        # print(x.size())
        x = x * ChannelAttn_whr(x)
        x = self.bottleneck_2(x)
        x = self.identity_block_2_1(x)
        x = self.identity_block_2_2(x)
        x = self.identity_block_2_3(x)
        # print(x.size())
        x = self.bottleneck_3(x)
        x = self.identity_block_3_1(x)
        x = self.identity_block_3_2(x)
        x = self.identity_block_3_3(x)
        x = self.identity_block_3_4(x)
        x = self.identity_block_3_5(x)
        # print(x.size())

        x = self.bottleneck_4(x)
        x = self.identity_block_4_1(x)
        x = self.identity_block_4_2(x)
        # print("feature shape:", x.size())

        if self.include_top:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            # TODO: why there is no dropout
            x = self.fc(x)
        return x


class Resnet50_CASA(nn.Module):
    def __init__(self, dropout_rate, num_classes, include_top):
        super(Resnet50_CASA, self).__init__()
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.include_top = include_top

        # Define the building blocks
        self.conv_3x3 = conv_1_3x3()

        self.bottleneck_1 = bottleneck(16, [16, 16, 64], kernel_size=3, strides=(1, 1))
        self.identity_block_1_1 = identity_block3(64, [16, 16, 64], kernel_size=3)
        self.identity_block_1_2 = identity_block3(64, [16, 16, 64], kernel_size=3)

        self.bottleneck_2 = bottleneck(64, [32, 32, 128], kernel_size=3, strides=(2, 2))
        self.identity_block_2_1 = identity_block3(128, [32, 32, 128], kernel_size=3)
        self.identity_block_2_2 = identity_block3(128, [32, 32, 128], kernel_size=3)
        self.identity_block_2_3 = identity_block3(128, [32, 32, 128], kernel_size=3)

        self.bottleneck_3 = bottleneck(128, [64, 64, 256], kernel_size=3, strides=(2, 2))
        self.identity_block_3_1 = identity_block3(256, [64, 64, 256], kernel_size=3)
        self.identity_block_3_2 = identity_block3(256, [64, 64, 256], kernel_size=3)
        self.identity_block_3_3 = identity_block3(256, [64, 64, 256], kernel_size=3)
        self.identity_block_3_4 = identity_block3(256, [64, 64, 256], kernel_size=3)
        self.identity_block_3_5 = identity_block3(256, [64, 64, 256], kernel_size=3)

        self.bottleneck_4 = bottleneck(256, [128, 128, 512], kernel_size=3, strides=(1, 1))
        self.identity_block_4_1 = identity_block3(512, [128, 128, 512], kernel_size=3)
        self.identity_block_4_2 = identity_block3(512, [128, 128, 512], kernel_size=3)

        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(512, num_classes)

        # Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_x):
        # print(input_x.size())
        x = self.conv_3x3(input_x)
        # print(x.size())
        x = self.bottleneck_1(x)
        x = self.identity_block_1_1(x)
        x = self.identity_block_1_2(x)
        # print(x.size())
        x = self.bottleneck_2(x)
        x = self.identity_block_2_1(x)
        x = self.identity_block_2_2(x)
        x = self.identity_block_2_3(x)
        xsa = SpatialAttn_whr(x)
        xca = ChannelAttn_whr(x)
        x = x * (xsa * xca)
        # print(x.size())
        x = self.bottleneck_3(x)
        x = self.identity_block_3_1(x)
        x = self.identity_block_3_2(x)
        x = self.identity_block_3_3(x)
        x = self.identity_block_3_4(x)
        x = self.identity_block_3_5(x)
        # print(x.size())
        x = self.bottleneck_4(x)
        x = self.identity_block_4_1(x)
        x = self.identity_block_4_2(x)
        # print("feature shape:", x.size())

        if self.include_top:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            # TODO: why there is no dropout
            x = self.fc(x)
        return x


# class Resnet34:
#     def __init__(self, training, dropout_rate, num_classes, reuse, include_top):
#         self.training = training
#         self.dropout_rate = dropout_rate
#         self.num_classes = num_classes
#         self.reuse = reuse
#         self.include_top = include_top
#
#     def conv_1_7x7(self, x, reuse):
#         x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=[7, 7], strides=2, padding='SAME',
#                              use_bias=False, name='conv_1_7x7', reuse=reuse)
#         x = tf.layers.batch_normalization(x, training=self.training, name='conv_1_7x7', reuse=reuse)
#         x = tf.nn.relu(x)
#         x = tf.layers.max_pooling2d(x, pool_size=3, strides=2, padding='valid', name='conv_1_7x7_pool')
#         return x
#
#     def forward(self, input_x):
#         x = self.conv_1_7x7(input_x, reuse=self.reuse)
#
#         x = basic_block(x, 3, [64, 64], stage=2, block='a', training=self.training, reuse=self.reuse, strides=(1, 1))
#         x = identity_block2(x, 3, [64, 64], stage=2, block='b', training=self.training, reuse=self.reuse)
#         x = identity_block2(x, 3, [64, 64], stage=2, block='c', training=self.training, reuse=self.reuse)
#
#         x = basic_block(x, 3, [128, 128], stage=3, block='a', training=self.training, reuse=self.reuse)
#         x = identity_block2(x, 3, [128, 128], stage=3, block='b', training=self.training, reuse=self.reuse)
#         x = identity_block2(x, 3, [128, 128], stage=3, block='c', training=self.training, reuse=self.reuse)
#         x = identity_block2(x, 3, [128, 128], stage=3, block='d', training=self.training, reuse=self.reuse)
#
#         x = basic_block(x, 3, [256, 256], stage=4, block='a', training=self.training, reuse=self.reuse)
#         x = identity_block2(x, 3, [256, 256], stage=4, block='b', training=self.training, reuse=self.reuse)
#         x = identity_block2(x, 3, [256, 256], stage=4, block='c', training=self.training, reuse=self.reuse)
#         x = identity_block2(x, 3, [256, 256], stage=4, block='d', training=self.training, reuse=self.reuse)
#         x = identity_block2(x, 3, [256, 256], stage=4, block='e', training=self.training, reuse=self.reuse)
#         x = identity_block2(x, 3, [256, 256], stage=4, block='f', training=self.training, reuse=self.reuse)
#
#         x = basic_block(x, 3, [512, 512], stage=5, block='a', training=self.training, reuse=self.reuse)
#         x = identity_block2(x, 3, [512, 512], stage=5, block='b', training=self.training, reuse=self.reuse)
#         x = identity_block2(x, 3, [512, 512], stage=5, block='c', training=self.training, reuse=self.reuse)
#
#         print("feature shape:", x.get_shape())
#
#         if self.include_top:
#             x = tf.reduce_mean(x, [1, 2])
#             x = tf.layers.dropout(x, rate=self.dropout_rate, training=self.training)
#             x = tf.layers.dense(inputs=x, use_bias=True, units=self.num_classes, name='linear', reuse=None)
#         return x
def resnet50(**kwargs):
    """
    Constructs a ResNet50 model.
    """
    return Resnet50(**kwargs)


def dconv_resnet50(**kwargs):
    """
    Constructs a DconvResnet50 model.
    """
    return DconvResnet50(**kwargs)


def sa_resnet50(**kwargs):
    """
    Constructs a Resnet50_SA model.
    """
    return Resnet50_SA(**kwargs)


def ca_resnet50(**kwargs):
    """
    Constructs a Resnet50_CA model.
    """
    return Resnet50_CA(**kwargs)


def casa_resnet50(**kwargs):
    """
    Constructs a Resnet50_SACA model.
    """
    return Resnet50_CASA(**kwargs)
