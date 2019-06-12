'''
Training script for CIFAR-10/100
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import argparse
import os
import itertools
import shutil

import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix
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
from offlineDA import MiniImageNet

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig


def main(args):
    global best_acc

    # Data loading code
    print('==> Preparing dataset')
    transform_test = transforms.Compose([
        # transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    valset = datasets.CIFAR100(args.data, train=False, transform=transform_test)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.test_batch, shuffle=False,
                                             num_workers=args.workers)
    # create model
    if args.arch.startswith('d1_resnet50'):
        model = models.__dict__[args.arch](
            num_classes=100,
            include_top=True,
            dropout_rate=0,
            layer=args.layer,
            is_shuff=False  # TODO: check
        )
    elif args.arch.endswith('vgg16_1d'):
        model = models.__dict__[args.arch](
            num_classes=100,
            include_top=True,
            dropout_rate=0,
            layer=args.layer,
            is_shuff=False
        )
    else:
        raise Exception('you should only choose vgg16_1d or d1_resnet50 as the model')

    model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = False
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    print('==> Resuming from checkpoint..')
    assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(args.resume)
    best_acc = checkpoint['best_acc']
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])

    print('\nEvaluation only')
    cm, diag = test(val_loader, model, criterion, start_epoch, use_cuda)
    print(diag)
    print(cm)
    np.set_printoptions(precision=2)

    plt.figure()
    plot_confusion_matrix(cm, classes=range(100),#['airplane', 'automobile','bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
                          name=args.layer, title='ResNet50 GAP+FC imgsize 32')


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
            outputs = model(inputs)
            _, pred = outputs.data.topk(1, 1, True, True)
            pred = pred.view(-1, )
            y_pred.extend(pred.cpu().tolist())
            y_true.extend(targets.data.cpu().tolist())
            # break
            bar.next()
        bar.finish()

    cnf_matrix = confusion_matrix(y_true, y_pred)

    return cnf_matrix, np.diag(cnf_matrix)


def plot_confusion_matrix(cm, classes, name,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, ' ', #format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(str(name) + '.png')


class Employee:
    pass


if __name__ == '__main__':
    args = Employee()  # Create an empty emp
    args.arch = 'd1_resnet50'
    args.layer = 99
    args.data = '/data/users/yuefan/fanyue/dconv/data/'
    args.gpu_id = None
    args.workers = 0
    args.test_batch = 100
    args.manualSeed = 6
    args.gpu_id = '1'
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

    # layers = [11, 12, 21, 22, 31, 32, 33, 41, 42, 43, 51, 52, 53, 99]
    # args.resume = '/data/users/yuefan/fanyue/dconv/checkpoints/cifar10/vgg161d_300_noDA/vgg161d_9953_300/model_best.pth.tar'
    # main(args)
    layers = [00, 10, 11, 12, 20, 21, 22, 23, 30, 31, 32, 33, 34, 35, 40, 41, 42, 99]
    # m = []
    for i in layers:
        args.layer = i
        if i == 0:
            args.resume = '/data/users/yuefan/fanyue/dconv/checkpoints/cifar100/resnet501d_300_noDA/resnet501d_0042_300/model_best.pth.tar'
        else:
            args.resume = '/data/users/yuefan/fanyue/dconv/checkpoints/cifar100/resnet501d_300_noDA/resnet501d_'+str(i)+'42_300/model_best.pth.tar'
            # '/BS/yfan/work/trained-models/dconv/checkpoints/mini-imagenet/resnet501d_300_imgsize64_noDA/resnet501d_'+str(i)+'42_300/model_best.pth.tar'
        print(args.resume)
        main(args)
        # break
        # m.append(main(args))
    # print(m)
    # import pickle
    # with open("resnet501d_mini_imgsize64_noDA.txt", "wb") as fp:  # Pickling
    #     pickle.dump(m, fp)
