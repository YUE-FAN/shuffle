import torchvision.models as m
from torchsummary import summary
import models.cifar as models
import torch.nn as nn


# a = m.resnet50()
# m.resnet50()
v = models.d1_resnet50(num_classes=10,
                        include_top=True,
                        dropout_rate=0,
                        layer=0,
                        is_shuff=False)
summary(v, (3,32,32))




