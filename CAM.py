'''
Training script for CIFAR-10/100
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import os
import itertools
import shutil

import matplotlib
import matplotlib.pyplot as plt
import time
import random
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn

import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.cifar as models
from offlineDA import MiniImageNet

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig


def main(args):
    global best_acc

    print('==> Preparing dataset')
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    if args.dataset == 'cifar10':
        num_class = 10
        valset = datasets.CIFAR10(root=args.dat, train=False, download=False, transform=transform_test)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=args.test_batch, shuffle=False,
                                                 num_workers=args.workers)
    elif args.dataset == 'cifar100':
        num_class = 100
        valset = datasets.CIFAR100(root=args.dat, train=False, download=False, transform=transform_test)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=args.test_batch, shuffle=False,
                                                 num_workers=args.workers)
    else:
        raise Exception('you should only choose cifar10 or cifar100')

    # create model
    if args.arch.startswith('resnet50'):
        model = models.__dict__[args.arch](
            num_classes=num_class,
            include_top=False,
            dropout_rate=0,
            layer=args.layer
        )
    elif args.arch.endswith('vgg16'):
        model = models.__dict__[args.arch](
            num_classes=num_class,
            include_top=False,
            dropout_rate=0,
            layer=args.layer
        )
    else:
        raise Exception('you should only choose vgg16_1d or d1_resnet50 as the model')

    model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = False
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    print('==> Resuming from checkpoint..')
    assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(args.resume)
    best_acc = checkpoint['best_acc']
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])

    print('\nEvaluation only')
    _, diag = test(val_loader, model, criterion, start_epoch, use_cuda)
    return diag.tolist()


def test(val_loader, model, criterion, epoch, use_cuda):
    global best_acc

    data_time = AverageMeter()

    # switch to evaluate mode
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        end = time.time()
        bar = Bar('Processing', max=len(val_loader))
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

            # compute output
            model.include_top = False
            outputs = model(inputs)  # [bs, num_feature, h, w]
            weights = model.fc.weight  # [num_class, num_feature]
            num_class, num_feature = weights.size()

            # [bs, num_class, h, w]
            cam = torch.empty(outputs.size(0), num_class, outputs.size(2), outputs.size(3)).cuda()
            for i in range(num_class):
                cam[:, i, :, :] = torch.matmul(outputs, weights[i].view(1, num_feature, 1, 1)).sum(dim=1)
            cam = cam[:, targets, :, :]  # select the ones corresponds to the correct label


            # select the ones which get predicted correctly
            model.include_top = True
            outputs = model(inputs)  # [bs, num_feature, h, w]
            _, pred = outputs.data.topk(1, 1, True, True)
            pred = pred.view(-1, )
            idx = pred.eq(targets)
            cam = cam[idx!=0]  # [bs, h, w]

            inputs = inputs[idx!=0]

            draw_CAM(cam, inputs)

            bar.next()
        bar.finish()

    return

def draw_CAM(cams, inputs):
    """
    
    :param cam: # [bs, h, w]
    :param inputs: # [bs, 3, h, w]
    :return: 
    """
    bs, _, h, w = inputs.size()
    for i in range(bs):
        cam = cams[i]
        img = inputs[i].transpose(1, 2, 0)



        cam = cv2.resize(cam, (64, 128), interpolation=cv2.INTER_CUBIC)
        cmap = matplotlib.cm.jet
        plt.imshow(cam, cmap=cmap)
        # plt.savefig(all_dir+'/{}/sa.jpg'.format(id))
        plt.savefig('./{}.jpg'.format(i))



class Employee:
    pass


if __name__ == '__main__':
    args = Employee()  # Create an empty emp
    args.dataset = 'cifar10'
    args.arch = 'resnet50'
    # args.data = '/BS/database11/mini-imagenet64/'
    args.data = '/data/users/yuefan/fanyue/dconv/data/'
    args.img_size = 32
    args.gpu_id = '0'
    args.workers = 0
    args.test_batch = 100
    args.manualSeed = 6

    # Use CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    use_cuda = torch.cuda.is_available()

    # Random seed
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.manualSeed)
        torch.backends.cudnn.deterministic = True
    best_acc = 0  # best test accuracy
    assert args.img_size == 32 or args.img_size == 64, "img size can only be 32 or 64!"

    # layers = [11, 12, 21, 22, 31, 32, 33, 41, 42, 43, 51, 52, 53, 99]
    layers = [00, 10, 11, 12, 20, 21, 22, 23, 30, 31, 32, 33, 34, 35, 40, 41, 41, 99]
    m = []
    for i in layers:
        args.layer = i
        if i == 0:
            args.resume = '/BS/yfan/work/trained-models/dconv/checkpoints/mini-imagenet/resnet501d_300_imgsize64_noDA/resnet501d_0042_300/model_best.pth.tar'
        else:
            args.resume = '/BS/yfan/work/trained-models/dconv/checkpoints/mini-imagenet/resnet501d_300_imgsize64_noDA/resnet501d_' + str(
                i) + '42_300/model_best.pth.tar'
        print(args.resume)
        m.append(main(args))
    print(m)
    import pickle

    with open("resnet501d_mini_imgsize64_noDA.txt", "wb") as fp:  # Pickling
        pickle.dump(m, fp)
