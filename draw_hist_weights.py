"""
This script draws the histogram of the weights of the ref model at different layers.
On one plot, there are 2 histogram, one corresponds to the center weights, the other corresponds to the surrounding weights

Examples:
python draw_hist_weights.py -d cifar100 -j 0 --dataset-path /BS/databases00/cifar-100 --manualSeed 6 --gpu-id 0
--refarch vgg16_shuffle --ref /BS/yfan/work/trained-models/dconv/checkpoints/cifar100/vgg16shuffleone_300_imgsize32_DA/vgg16shuffleone_99_300/
"""

from __future__ import print_function

import argparse
import os
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from sklearn.metrics import confusion_matrix
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.cifar as models
from models.cifar import Resnet50_1x1LAP, VGG16_1x1LMP, VGG16_Shuffle
from models.cifar.resnetfy import bottleneck, identity_block3
from models.cifar.vggfy import CONV_3x3
import matplotlib.pyplot as plt

from utils import Bar, AverageMeter, accuracy

# ['alexnet', 'bottleneck', 'conv_1_7x7', 'densenet', 'identity_block3', 'preresnet', 'resnet', 'resnext', 'vgg11',
# 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'wrn']
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('--dataset-path', type=str)
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
# Checkpoints
parser.add_argument('--ref', default='', type=str, metavar='PATH',
                    help='path to ref model (default: none)')
parser.add_argument('--refarch', type=str, metavar='ARCH')
# Device options
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
np.random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
    torch.backends.cudnn.deterministic = True


def create_model(arch, num_classes, layer):
    model = models.__dict__[arch](
        num_classes=num_classes,
        include_top=True,
        dropout_rate=0,
        layer=layer
    )
    return model


def create_dataloader(dataset_type, dataset_path):
    assert dataset_type in ['cifar10', 'cifar100', 'imagenet'], 'Dataset can only be cifar10 or cifar100 or imagenet'
    if dataset_type == 'cifar10':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        num_classes = 10
        testset = datasets.CIFAR10(root=dataset_path, train=False, download=False, transform=transform_test)
        testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    elif dataset_type == 'cifar100':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        num_classes = 100
        testset = datasets.CIFAR100(root=dataset_path, train=False, download=False, transform=transform_test)
        testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    else:
        num_classes = 1000
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        testset = datasets.ImageFolder(dataset_path, transform_test)
        testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers, pin_memory=True)

    return num_classes, testloader


def ploter(weight, save_path):
    weight = weight.view(weight.size(0) * weight.size(1), 9)
    center = weight[:, 4].view(-1)
    surround = weight[:, torch.arange(9) != 4].view(-1)
    fig, axs = plt.subplots(1, 1)
    axs.hist(center, bins=200, label='center', density=True, fc=(0, 0, 1, 0.5))
    axs.hist(surround, bins=200, label='surround', density=True, fc=(1, 0, 0, 0.5))
    axs.legend(loc='upper right')
    plt.savefig(save_path)


def main():
    if args.refarch.startswith('vgg16'):
        layer_list = [11, 12, 21, 22, 31, 32, 33, 41, 42, 43, 51, 52, 53]
    elif args.refarch.startswith('resnet50'):
        layer_list = [00, 10, 11, 12, 20, 21, 22, 23, 30, 31, 32, 33, 34, 35, 40, 41, 42]
    else:
        raise Exception('invalid args.arch')
    # Data
    print('==> Preparing dataset %s' % args.dataset)
    num_classes, testloader = create_dataloader(args.dataset, args.dataset_path)

    # ref model
    refmodel = create_model(args.refarch, num_classes, 99)
    refmodel = torch.nn.DataParallel(refmodel).cuda()
    cudnn.benchmark = False
    refpp = os.path.join(args.ref, 'model_best.pth.tar')
    assert os.path.isfile(refpp), 'Error: no ref checkpoint directory found!'
    refcheckpoint = torch.load(refpp)
    refmodel.load_state_dict(refcheckpoint['state_dict'])  # sanity check
    # _, refperclass_acc, reftest_acc = test(testloader, refmodel, use_cuda)
    # print('ref acc is', reftest_acc)

    if args.refarch.startswith('vgg16'):
        for i, m in enumerate(refmodel.module.children()):
            if isinstance(m, CONV_3x3):
                print(layer_list[i])
                weight = m.conv.weight.detach().cpu()
                ploter(weight, str(layer_list[i])+'del.png')
            elif isinstance(m, nn.Sequential) and len(m) == 2:
                print(layer_list[i])
                weight = m[0].conv.weight.detach().cpu()
                ploter(weight, str(layer_list[i])+'del.png')
    elif args.refarch.startswith('resnet50'):
        for i, m in enumerate(refmodel.module.children()):
            if isinstance(m, identity_block3) or isinstance(m, bottleneck):
                print(layer_list[i])
                weight = m.conv2.weight.detach().cpu()
                ploter(weight, str(layer_list[i])+'del.png')

    else:
        raise Exception('invalid args.arch')
    return


def test(val_loader, model, use_cuda):
    data_time = AverageMeter()
    top1 = AverageMeter()
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
            outputs = model(inputs)

            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            top1.update(prec1.item(), inputs.size(0))

            _, pred = outputs.data.topk(1, 1, True, True)
            pred = pred.view(-1, )
            y_pred.extend(pred.cpu().tolist())
            y_true.extend(targets.data.cpu().tolist())
            bar.next()
        bar.finish()

    cnf_matrix = confusion_matrix(y_true, y_pred)
    return cnf_matrix, np.diag(cnf_matrix), top1.avg


if __name__ == '__main__':
    main()
