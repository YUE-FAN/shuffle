"""
This code read from 11-99 models, and read out their best test acc from their log.txts
"""

import os
import numpy as np
path = '/data/users/yuefan/fanyue/dconv/checkpoints/fashion-mnist/alexnet1d_300_DA/'

final = []
dir_list = np.sort(os.listdir(path))
for i in dir_list:
    if i == 'README.md':
        continue
    # print(path+i+'/log.txt')
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


# from torchvision.models import vgg16_bn, resnet50
# import torch
# from torch.autograd import Variable
# import numpy as np
# 
# 
# def print_model_parm_flops(model, input_size, multiply_adds):
#     # prods = {}
#     # def save_prods(self, input, output):
#     # print 'flops:{}'.format(self.__class__.__name__)
#     # print 'input:{}'.format(input)
#     # print '_dim:{}'.format(input[0].dim())
#     # print 'input_shape:{}'.format(np.prod(input[0].shape))
#     # grads.append(np.prod(input[0].shape))
# 
#     prods = {}
# 
#     def save_hook(name):
#         def hook_per(self, input, output):
#             # print 'flops:{}'.format(self.__class__.__name__)
#             # print 'input:{}'.format(input)
#             # print '_dim:{}'.format(input[0].dim())
#             # print 'input_shape:{}'.format(np.prod(input[0].shape))
#             # prods.append(np.prod(input[0].shape))
#             prods[name] = np.prod(input[0].shape)
#             # prods.append(np.prod(input[0].shape))
# 
#         return hook_per
# 
#     list_1 = []
# 
#     def simple_hook(self, input, output):
#         list_1.append(np.prod(input[0].shape))
# 
#     list_2 = {}
# 
#     def simple_hook2(self, input, output):
#         list_2['names'] = np.prod(input[0].shape)
# 
#     list_conv = []
# 
#     def conv_hook(self, input, output):
#         batch_size, input_channels, input_height, input_width = input[0].size()
# 
#         output_channels, output_height, output_width = output[0].size()
# 
#         kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups) * (
#         2 if multiply_adds else 1)
#         bias_ops = 1 if self.bias is not None else 0
# 
#         params = output_channels * (kernel_ops + bias_ops)
#         flops = batch_size * params * output_height * output_width
# 
#         list_conv.append(flops)
# 
#     list_linear = []
# 
#     def linear_hook(self, input, output):
#         batch_size = input[0].size(0) if input[0].dim() == 2 else 1
# 
#         weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
# 
#         if self.bias is None:
#             bias_ops = 0
#         else:
#             bias_ops = self.bias.nelement()
# 
#         flops = batch_size * (weight_ops + bias_ops)
#         list_linear.append(flops)
# 
#     list_bn = []
# 
#     def bn_hook(self, input, output):
#         list_bn.append(4*input[0].nelement())
# 
#     list_relu = []
# 
#     def relu_hook(self, input, output):
#         list_relu.append(input[0].nelement())
# 
#     list_pooling = []
# 
#     def pooling_hook(self, input, output):
#         batch_size, input_channels, input_height, input_width = input[0].size()
#         output_channels, output_height, output_width = output[0].size()
# 
#         kernel_ops = self.kernel_size * self.kernel_size
#         bias_ops = 0
#         params = output_channels * (kernel_ops + bias_ops)
#         flops = batch_size * params * output_height * output_width
# 
#         list_pooling.append(flops)
# 
#     def foo(net):
#         childrens = list(net.children())
#         if not childrens:
#             if isinstance(net, torch.nn.Conv2d):
#                 # net.register_forward_hook(save_hook(net.__class__.__name__))
#                 # net.register_forward_hook(simple_hook)
#                 # net.register_forward_hook(simple_hook2)
#                 net.register_forward_hook(conv_hook)
#             if isinstance(net, torch.nn.Linear):
#                 net.register_forward_hook(linear_hook)
#             if isinstance(net, torch.nn.BatchNorm2d):
#                 net.register_forward_hook(bn_hook)
#             if isinstance(net, torch.nn.ReLU):
#                 net.register_forward_hook(relu_hook)
#             if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
#                 net.register_forward_hook(pooling_hook)
#             return
#         for c in childrens:
#             foo(c)
# 
#     resnet = model
#     resnet.eval()
#     foo(resnet)
#     input = Variable(torch.rand(input_size).unsqueeze(0), requires_grad=True)
#     out = resnet(input)
# 
#     total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling))
# 
#     print('  + Number of FLOPs: %.2fM' % (total_flops / 1e6))
# 
#     # print list_bn
# 
# 
#     # print 'prods:{}'.format(prods)
#     # print 'list_1:{}'.format(list_1)
#     # print 'list_2:{}'.format(list_2)
#     # print 'list_final:{}'.format(list_final)
# 
# a = resnet50()
# print_model_parm_flops(a, (3, 32, 32), False)
# print('    Total params: %.2fM' % (sum(p.numel() for p in a.parameters())/1000000.0))

