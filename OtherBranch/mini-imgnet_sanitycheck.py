# sanity check for my own mini-imagenet dataset. You should change the var [present_class] and see if
# the training set and test set provides images belonging to the same class.
# Furthermore, you could change [trainset] and [testset] see if they are aligned.

from offlineDA import MiniImageNet
import torchvision.transforms as transforms
import torch.utils.data as data
import torch
from PIL import Image
import numpy as np

transform = transforms.Compose([transforms.ToTensor()])

trainset = MiniImageNet('/BS/database11/mini-imagenet64/', 64, True, transform=transform)
trainloader = data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=32)


testset = MiniImageNet('/BS/database11/mini-imagenet64/', 64, False, transform=transform)
testloader = data.DataLoader(testset, batch_size=100, shuffle=True, num_workers=32)

present_class = 77

# choose images from training set
for idx, (inputs, targets) in enumerate(trainloader):
    inputs = inputs*255
    inputs = inputs.to(torch.uint8).numpy()

    print(inputs.shape)
    print(targets)

    for i, tar in enumerate(targets):
        if tar == present_class:
            a = np.transpose(inputs[i], (1, 2, 0))
            im = Image.fromarray(a)
            im.show()

    if idx == 5:
         break

# choose images from test set
for idx, (inputs, targets) in enumerate(testloader):
    inputs = inputs*255
    inputs = inputs.to(torch.uint8).numpy()

    print(inputs.shape)
    print(targets)

    for i, tar in enumerate(targets):
        if tar == present_class:
            a = np.transpose(inputs[i], (1, 2, 0))
            im = Image.fromarray(a)
            im.show()

    if idx == 5:
         break
