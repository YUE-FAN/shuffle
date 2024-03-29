"""
Training script for MNIST

10 classes
Images are 28x28 pixels, grey scale.
6000 training images, 1000 test images per class

Usage:
python mnist.py --dataset mnist --depth 110 --epochs 60 --schedule 50 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/mnist/mlp_60 --gpu-id 0
"""

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
import models.mnist as models

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

# ['alexnet', 'bottleneck', 'conv_1_7x7', 'densenet', 'identity_block3', 'preresnet', 'resnet', 'resnext', 'vgg11',
# 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'wrn']
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
# Datasets
parser.add_argument('-d', '--dataset', default='mnist', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=50, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4 on cifar)')
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
parser.add_argument('--DA', action='store_true')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')  # 0,1

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.dataset == 'mnist'

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy


# class Linear_Model(nn.Module):
#     def __init__(self, inplane, num_class):
#         super(Linear_Model, self).__init__()
#         self.fc = nn.Conv2d(1, num_class, kernel_size=28, stride=1, padding=0, bias=False)
#
#     def forward(self, input):
#         # np.save('/nethome/yuefan/fanyue/dconv/l_w.npy', self.fc.weight.detach().cpu().numpy())
#         # np.save('/nethome/yuefan/fanyue/dconv/l_b.npy', self.fc.bias.detach().cpu().numpy())
#         x = self.fc(input)
#         x = x.view(x.size(0), -1)
#         return x
#
#
# class MLP(nn.Module):
#     def __init__(self, num_hidden, num_class):
#         super(MLP, self).__init__()
#         self.hidden = nn.Conv2d(1, num_hidden, kernel_size=28, stride=1, padding=0, bias=False)
#         self.relu = nn.ReLU(inplace=True)
#         self.fc = nn.Linear(num_hidden, num_class, bias=False)
#         print('number of hidden layers: ', num_hidden)
#
#     def forward(self, input):
#         x = self.hidden(input)
#         x = self.relu(x)
#         x_shape = x.size()  # [128, 3, 32, 32]
#         shuffled_input = torch.empty(x_shape[0], x_shape[1], x_shape[2], x_shape[3]).cuda(0)
#         perm1 = torch.randperm(x_shape[1])
#         shuffled_input[:, :, :, :] = x[:, perm1, :, :]
#         shuffled_input = shuffled_input.view(shuffled_input.size(0), -1)
#         x = self.fc(shuffled_input)
#         return x


def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Data
    print('==> Preparing dataset %s' % args.dataset)
    if args.DA:
        print('use DA')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),  # with p = 1
            transforms.RandomHorizontalFlip(),  # with p = 0.5
            transforms.ToTensor(),  # it must be this guy that makes it CHW again
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
    else:
        print('no DA')
        transform_train = transforms.Compose([
            transforms.ToTensor(),  # it must be this guy that makes it CHW again
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    dataloader = datasets.MNIST
    num_classes = 10

    trainset = dataloader(root='/data/users/yuefan/fanyue/dconv/data/mnist', train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

    testset = dataloader(root='/data/users/yuefan/fanyue/dconv/data/mnist', train=False, download=True, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    # Model
    print("==> creating model '{}'".format(args.arch))

    # model = Linear_Model(28*28, num_classes)
    # model = MLP(args.width, num_classes)
    if args.arch.endswith('resnet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                )
    elif args.arch.endswith('resnet50'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    include_top=True,
                    dropout_rate=0
                )
    elif args.arch.startswith('vgg'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    include_top=True,
                    dropout_rate=0
                )
    elif args.arch.startswith('alexnet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    layer=args.layer
                )
    else:
        model = models.__dict__[args.arch](num_classes=num_classes)
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    for name, param in model.named_parameters():
        print(name)
    # for name in model.named_modules():
    #     print(name)
    # for param in model.parameters():
    #     print(param)

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Resume
    title = 'mnist-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(testloader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch, use_cuda)
        test_loss, test_acc = test(testloader, model, criterion, epoch, use_cuda)

        # append logger file
        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint)

    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best acc:')
    print(best_acc)


def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # print(type(inputs), type(targets))
        # print(inputs.size(), len(targets))
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(async=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
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
    # avg_img = np.zeros(shape=(28, 28))
    # cc = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # for i in range(len(targets)):
        #     if targets[i] == 9:
        #         img = inputs[i, 0, :, :]
        #         avg_img += img.detach().cpu().numpy()
        #         cc += 1

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
    # np.save('/nethome/yuefan/fanyue/dconv/img9.npy', avg_img / cc)
    return (losses.avg, top1.avg)


# def test_drop(testloader, model, criterion, epoch, use_cuda):
#     # this function is for random delete and selective delete, it can produce the test acc for each class
#     global best_acc
#
#     batch_time = AverageMeter()
#     data_time = AverageMeter()
#     losses = AverageMeter()
#     top1 = AverageMeter()
#     top5 = AverageMeter()
#
#     # switch to evaluate mode
#     model.eval()
#
#     end = time.time()
#     # bar = Bar('Processing', max=len(testloader))
#     class_top1 = np.zeros((10, ))
#     for batch_idx, (inputs, targets) in enumerate(testloader):
#         # measure data loading time
#         data_time.update(time.time() - end)
#
#         if use_cuda:
#             inputs, targets = inputs.cuda(), targets.cuda()
#         inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)
#
#         # compute output
#         outputs = model(inputs)
#         loss = criterion(outputs, targets)
#
#         _, pred = outputs.data.topk(1, 1, True, True)
#         pred = pred.t()
#         gt = targets.data.view(1, -1).expand_as(pred)
#         correct = pred.eq(gt)
#
#         for i in range(10):
#             class_correct = correct[gt == i]
#             # print(class_correct)
#             # print(class_correct.size())
#             if class_correct.size(0) == 0:
#                 continue
#             class_top1[i] += class_correct.view(-1).float().sum(0)
#
#
#         # measure accuracy and record loss
#         # prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
#         losses.update(loss.data[0], inputs.size(0))
#         # top1.update(prec1[0], inputs.size(0))
#         # top5.update(prec5[0], inputs.size(0))
#
#         # measure elapsed time
#         batch_time.update(time.time() - end)
#         end = time.time()
#     print(class_top1 / 800)
#     print(np.sum(class_top1)/8000)
#         # plot progress
#     #     bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
#     #                 batch=batch_idx + 1,
#     #                 size=len(testloader),
#     #                 data=data_time.avg,
#     #                 bt=batch_time.avg,
#     #                 total=bar.elapsed_td,
#     #                 eta=bar.eta_td,
#     #                 loss=losses.avg,
#     #                 top1=top1.avg,
#     #                 top5=top5.avg,
#     #                 )
#     #     bar.next()
#     # bar.finish()
#
#     return (losses.avg, top1.avg)


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']


if __name__ == '__main__':
    main()
