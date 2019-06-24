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
