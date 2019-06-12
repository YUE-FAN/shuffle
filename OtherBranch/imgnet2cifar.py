'''
This code resizes the full ImageNet dataset and saves it in the cifar format.

Training data is first center cropped into 224x224. then it is resized into 32x32 or 64x64.

Test data is processed in the same way.

'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.cifar as models

from torchvision.datasets import DatasetFolder
import pickle



from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig




IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

from PIL import Image

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class ppgg(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(self, root, transform=None, target_transform=None,
                 loader=pil_loader):
        super(ppgg, self).__init__(root, loader, IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform)
        self.imgs = self.samples

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target


def main():
    img_size = 64
    bs = 256
    workers = 32
    # Data loading code
    traindir = '/BS/database11/ILSVRC2012/train/'

    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Resize(img_size),
        transforms.ToTensor()
    ])

    train_loader = torch.utils.data.DataLoader(
        ppgg(traindir, transform_train),
        batch_size=bs, shuffle=True,
        num_workers=workers)

    train(train_loader, 1, 1, 1, 1, 1)


def train(train_loader, model, criterion, optimizer, epoch, use_cuda):
    bar = Bar('Processing', max=len(train_loader))
    pdata = dict()
    pdata['data'] = []
    pdata['labels'] = []
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # print(type(inputs), type(targets))
        # print(inputs, targets)
        inputs = inputs*255
        inputs = inputs.to(torch.uint8).numpy()
        pdata['data'].extend(inputs)

        pdata['labels'].extend(targets.numpy())
        bar.next()
    bar.finish()
    pdata['data'] = np.asarray(pdata['data'], dtype=np.uint8)
    # print(pdata['data'].shape)
    # print(pdata)
    # from PIL import Image

    # x = pdata['data'][1]
    # x = np.transpose(x, (1, 2, 0))
    # im = Image.fromarray(x)
    # im.show()
    pdata1 = dict()
    pdata1['data'] = pdata['data'][:250000]
    pdata1['labels'] = pdata['labels'][:250000]

    pdata2 = dict()
    pdata2['data'] = pdata['data'][250000:250000*2]
    pdata2['labels'] = pdata['labels'][250000:250000*2]

    pdata3 = dict()
    pdata3['data'] = pdata['data'][250000*2:250000*3]
    pdata3['labels'] = pdata['labels'][250000*2:250000*3]

    pdata4 = dict()
    pdata4['data'] = pdata['data'][250000*3:250000*4]
    pdata4['labels'] = pdata['labels'][250000*3:250000*4]

    pdata5 = dict()
    pdata5['data'] = pdata['data'][250000 * 4:]
    pdata5['labels'] = pdata['labels'][250000 * 4:]

    with open(r"/BS/database11/ILSVRC2012_imgsize64/train0", "wb") as output_file:
        pickle.dump(pdata1, output_file)
    with open(r"/BS/database11/ILSVRC2012_imgsize64/train1", "wb") as output_file:
        pickle.dump(pdata2, output_file)
    with open(r"/BS/database11/ILSVRC2012_imgsize64/train2", "wb") as output_file:
        pickle.dump(pdata3, output_file)
    with open(r"/BS/database11/ILSVRC2012_imgsize64/train3", "wb") as output_file:
        pickle.dump(pdata4, output_file)
    with open(r"/BS/database11/ILSVRC2012_imgsize64/train4", "wb") as output_file:
        pickle.dump(pdata5, output_file)
    return



if __name__ == '__main__':
    main()

