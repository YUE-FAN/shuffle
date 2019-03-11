'''
This file automates the testing procedure of the shuffle experiments
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

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

# ['alexnet', 'bottleneck', 'conv_1_7x7', 'densenet', 'identity_block3', 'preresnet', 'resnet', 'resnext', 'vgg11',
# 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'wrn']
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

# Use CUDA
use_cuda = torch.cuda.is_available()

# Random seed

manualSeed = random.randint(1, 10000)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(manualSeed)

best_acc = 0  # best test accuracy


def main():
    global best_acc

    dataset = 'cifar10'
    workers = 4
    test_batch = 100
    arch = 'resnet50'

    # Data
    print('==> Preparing dataset %s' % dataset)
    transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),  # with p = 1
        # transforms.RandomHorizontalFlip(),  # with p = 0.5
        transforms.ToTensor(),  # it must be this guy that makes it CHW again
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100

    testset = dataloader(root='/data/users/yuefan/fanyue/dconv/data', train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=test_batch, shuffle=False, num_workers=workers)

    # m_list = ['shuffleall', 'shuffle', 'rand', 'cshuffle', 'crand', 'none']
    m_list = ['crand']
    for m in m_list:  # for each trained model
        resume = '/data/users/yuefan/fanyue/dconv/checkpoints/'+dataset+'/dconv_' + m + '_'+arch+'_4242_winu_60/model_best.pth.tar'
        print(resume)
        for dconv_type in m_list:
            # Model
            model = models.__dict__[arch](
                num_classes=num_classes,
                include_top=True,
                dropout_rate=0,
                type=dconv_type
            )

            model = torch.nn.DataParallel(model).cuda()
            cudnn.benchmark = True
            # print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

            criterion = nn.CrossEntropyLoss()

            # Resume

            # Load checkpoint.
            # print('==> Resuming from checkpoint..')
            assert os.path.isfile(resume), 'Error: no checkpoint directory found!'
            checkpoint = torch.load(resume)
            best_acc = checkpoint['best_acc']
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])

            a = []
            for ii in range(10):  # for each configuration, test 10 times
                test_loss, test_acc = test(testloader, model, criterion, start_epoch, use_cuda)
                print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
                a.append(test_acc)
            print("the mean of ", m, " for ", dconv_type," is ", np.mean(a))


def test(testloader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)


if __name__ == '__main__':
    main()
