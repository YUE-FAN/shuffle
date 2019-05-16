'''
Training script for CIFAR-10/100
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import argparse
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.cifar as models
from ERF.erf_computer import erf_computer


# ['alexnet', 'bottleneck', 'conv_1_7x7', 'densenet', 'identity_block3', 'preresnet', 'resnet', 'resnext', 'vgg11',
# 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'wrn']
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--layer', type=int)
parser.add_argument('--shuff', type=int)
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')  # 0,1

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()


def main():
    # Data
    print('==> Preparing dataset %s' % args.dataset)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100

    testset = dataloader(root='/data/users/yuefan/fanyue/dconv/data', train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=args.workers)

    # Model
    print("==> creating model '{}'".format(args.arch))

    if args.arch.startswith('d1_resnet50'):
        model = models.__dict__[args.arch](
            num_classes=num_classes,
            include_top=True,
            dropout_rate=0,
            layer=args.layer,
            is_shuff=False  # TODO: check
        )
    elif args.arch.endswith('vgg16'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    include_top=True,
                    dropout_rate=0,
                    layer=args.layer
                )
    elif args.arch.endswith('vgg16_sa'):
        model = models.__dict__[args.arch](
            num_classes=num_classes,
            include_top=True,
            dropout_rate=0,
            layer=args.layer
        )
    elif args.arch.endswith('vgg16_1d'):
        model = models.__dict__[args.arch](
            num_classes=num_classes,
            include_top=True,
            dropout_rate=0,
            layer=args.layer,
            is_shuff=False
        )

    model = torch.nn.DataParallel(model).cuda()
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # for name, param in model.named_parameters():
    #     print(name)
    # for name in model.named_modules():
    #     print(name)
    # for param in model.parameters():
    #     print(param)

    # Resume
    title = 'cifar-10-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    bb = []
    for batch_idx, (inputs, targets) in enumerate(testloader):

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        bb.extend(erf_computer(model, inputs, 2))

    bb = np.array(bb)
    print(bb.mean())
    print(bb.std())
    print(np.median(bb))
    # print(bb/10000*100)

    print('Best acc:')
    print(best_acc)




if __name__ == '__main__':
    main()
