'''
For a given model family and a dataset: (resnet1d_3342, cifar100)
This script evaluates the manifold distance between each test image and the training set.
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
import faiss

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from scipy.spatial import KDTree

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


def main():
    dataset = 'cifar100'
    arch = 'resnet50'
    model_type = 'resnet50_shuffle_bad_1142'
    layer = 11

    workers = 4
    test_batch = 100

    print('==> Preparing dataset %s' % dataset)
    tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    elif dataset == 'cifar100':
        dataloader = datasets.CIFAR100
        num_classes = 100
    elif dataset == 'svhn':
        dataloader = datasets.SVHN
        num_classes = 10
    else:
        raise Exception('Only support CIFAR and SVHN!!!')

    trainset = dataloader(root='/data/users/yuefan/fanyue/dconv/data', train=True, download=True,
                          transform=tfms)
    trainloader = data.DataLoader(trainset, batch_size=test_batch, shuffle=True, num_workers=workers)
    testset = dataloader(root='/data/users/yuefan/fanyue/dconv/data', train=False, download=True,
                         transform=tfms)
    testloader = data.DataLoader(testset, batch_size=test_batch, shuffle=False, num_workers=workers)

    model_nums = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    for model_num in model_nums:
        resume = '/data/users/yuefan/fanyue/dconv/checkpoints/' + dataset + '/' + model_type + '_' + str(model_num) + '_60/model_best.pth.tar'
        print(resume)

        if arch.startswith('resnet50'):
            model = models.__dict__[arch](
                num_classes=num_classes,
                include_top=False,
                dropout_rate=0,
                layer=layer
            )
        elif arch.startswith('d1_resnet50'):
            model = models.__dict__[arch](
                num_classes=num_classes,
                include_top=False,
                dropout_rate=0,
                layer=layer,
                is_shuff=False  # TODO: check
            )
        else:
            raise Exception('The arch is not supported!')
        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True
        assert os.path.isfile(resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['state_dict'])

        saved_path = '/data/users/yuefan/fanyue/dconv/maruis_conjecture/'+dataset+'/'+model_type+'_'+str(model_num)+'/'
        if not os.path.isdir(saved_path):
            mkdir_p(saved_path)

        test_data_represent_list = test(testloader, model, use_cuda, 10000)  # 335MB
        test_data_represent_list = np.array(test_data_represent_list)
        train_data_represent_list = test(trainloader, model, use_cuda, 50000)  # 1.6GB
        train_data_represent_list = np.array(train_data_represent_list)

        index = faiss.IndexFlatL2(512)  # build the index IndexFlatIP
        print(index.is_trained)
        index.add(train_data_represent_list)  # add vectors to the index
        print(index.ntotal)

        k = 5  # we want to see 5 nearest neighbors
        D, I = index.search(train_data_represent_list, k)  # actual search
        np.save(saved_path + 'train_img_manidist_I.npy', I)
        np.save(saved_path + 'train_img_manidist_D.npy', D)  # TODO: note D is the squared euclidean distance
        
        # # np.save(save_path+'test.npy', test_data_represent_list)
        #
        
        # # np.save(save_path + 'train.npy', train_data_represent_list)
        #
        # # compute the manifold distance for each of the test image
        # del model
        # tree = KDTree(train_data_represent_list)
        # distances, _ = tree.query(test_data_represent_list, k=5, p=2)
        # distances = np.array(distances).mean(axis=1)
        # np.save(save_path + 'test_img_manidist_list.npy', distances)


def test(testloader, model, use_cuda, loader_len):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    data_represent_list = []

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        bar = Bar('Processing', max=len(testloader))
        for batch_idx, (inputs, targets) in enumerate(testloader):
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

            # compute output
            outputs = model(inputs)
            outputs = torch.nn.functional.avg_pool2d(outputs, kernel_size=(4, 4), stride=(1, 1))
            outputs = outputs.view(outputs.size(0), -1)

            data_represent_list.extend(outputs.detach().cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:}'.format(
                        batch=batch_idx + 1,
                        size=len(testloader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td
                        )
            bar.next()
        bar.finish()
    return data_represent_list


if __name__ == '__main__':
    main()
