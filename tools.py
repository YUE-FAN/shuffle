"""
This code read from 11-99 models, and read out their best test acc from their log.txts
"""

import os
import numpy as np
path = '/data/users/yuefan/fanyue/dconv/checkpoints/cifar100/resnet501d_300/'
final = []
dir_list = np.sort(os.listdir(path))
for i in dir_list:
    print(path+i+'/log.txt')
    # if i == 'vgg161d_9953_60':
    #     continue
    with open(path+i+'/log.txt', 'r') as f:
        l = f.readlines()
        l = l[1:]
        a = []
        for ll in l:
            tmp = ll.split()
            a.append(float(tmp[4]))
        print(max(a))
        final.append(max(a))
