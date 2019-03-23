# import torch
# import torch.nn as nn
#
# a = 1
# m = nn.Conv2d(16, 33, a, stride=1, padding=(a-1)/2)
# input = torch.randn(20, 16, 99, 99)
# print(input.size())
# output = m(input)
# print(output.size())







# from bokeh.io import output_file, show
# from bokeh.layouts import gridplot
# from bokeh.models import ColumnDataSource
# from bokeh.plotting import figure
#
# output_file("brushing.html")
#
# x =  [0,0,0,0,1,1]
# y =  [0,0,0,0,1,1]
# z1 = [0,0,0,0,1,1]
# z2 = [0,1,2,3,0,1]
#
# # create a column data source for the plots to share
# source = ColumnDataSource(data=dict(x=x, y=y, z1=z1, z2=z2))
#
# TOOLS = "box_select,lasso_select,help"
#
# # create a new plot and add a renderer
# left = figure(tools=TOOLS, plot_width=300, plot_height=300, title=None)
# left.circle('x', 'y', source=source)
#
# # create another new plot and add a renderer
# right = figure(tools=TOOLS, plot_width=300, plot_height=300, title=None)
# right.circle('z1', 'z2', source=source)
#
# s = figure(tools=TOOLS, plot_width=300, plot_height=300, title=None)
# s.circle('z1', 'z2', source=source)
#
# p = gridplot([[left, right, s]])
#
# show(p)



# import torch
# import torch.nn as nn
# x = torch.tensor([[1,2,3], [4,5,6], [7,8,9], [10,11,12]], dtype= torch.float)  # [[[10,20,30], [40,50,60]], [[70,80,90], [100,110,120]],
# x_shape = x.size()  # 233
# y = x.clone().view(-1, int(x_shape[1]), int(x_shape[2]), 2)
# y.data.fill_(5)
# print(x.size())
# print(x)
# print(y.size())
# print(x)
# print(x.size())  # [1, 2, 2, 3]
# print(x[0,0,1,2])
# x = x.view(1, 2, 6)
# print(x)
# print(x[0,0,5])
# print(5//3, 5%3)


# import numpy as np
# from PIL import Image
# a = np.load('/nethome/yuefan/fanyue/dconv/fm3x3.npy')  # img, fm3x3, fmbottel4
# print(a.shape)
# a = a.reshape(16, 32, 32)
# a = a.transpose(1,2,0)
#
# # b = (a*255).astype('uint8')
# # im = Image.fromarray(b) # monochromatic image
# # im.show()
# b = a[:, :, 0]  # 0 and 14
# b = (b*255).astype('uint8')
# im = Image.fromarray(b) # monochromatic image
# im.show()

# import torch.nn as nn
# import torch
# # non-square kernels and unequal stride and with padding
# m = nn.Conv3d(20, 14, (3, 3, 3), stride=(1, 2, 1), padding=(1, 1, 1))
# input = torch.randn(1, 20, 4, 4, 4)
# output = m(input)
# print(input.size())
# print(output.size())

# import torch.nn as nn
# import torch
# x = torch.arange(0, 96).view(1, 6, 4, 4)
# a = nn.Conv3d(1, 2, kernel_size=(5,3,3), stride=1, padding=(2,1,1), bias=False)
# print(x.size())
#
# # a.weight.data = torch.arange(0, 27).view(1, 3, 3, 3)
# # nn.init.constant_(a.weight, 1)
# # print(a.weight.data.size())
# # o = a(x)
# # print(o.size())
# # print(o)
#
# m = nn.MaxPool2d(1, stride=2)
#
# o = m(x)
#
# print(o.size())

# How to freeze layers
# for param in model.parameters():
#     if list(param.size()) == [10, 512] or list(param.size()) == [10]:
#         print(param.size())
#         param.requires_grad = True
#     else:
#         param.requires_grad = False
#
# model = torch.nn.DataParallel(model).cuda()
# cudnn.benchmark = False  # TODO: for deterministc result, this has to be false
# print('    Total params: %.2fM' % (
#             sum(p.numel() for p in filter(lambda p: p.requires_grad, model.parameters())) / 1000000.0))
import numpy as np


def loader(dataset, model_type, is_train):
    model_nums = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    mm = []
    for model_num in model_nums:
        saved_path = '/data/users/yuefan/fanyue/dconv/maruis_conjecture/' + dataset + '/' + model_type + '_' + str(
            model_num) + '/'
        if is_train:
            D = np.load(saved_path + 'train_img_manidist_D.npy')
        else:
            D = np.load(saved_path + 'test_img_manidist_D.npy')
        D = np.sqrt(D)
        tmp = D.mean(axis=1)
        mm.append(tmp)
    mm = np.array(mm)
    return np.mean(mm, axis=0)


dataset = 'cifar100'

model_type = 'resnet50'
train_ref = loader(dataset, model_type, True)
model_type = 'resnet501d_good_3042'
train_good = loader(dataset, model_type, True)
model_type = 'resnet50_shuffle_good_3342'
train_shuffle_good = loader(dataset, model_type, True)
model_type = 'resnet501d_bad_1142'
train_bad = loader(dataset, model_type, True)

model_type = 'resnet50'
test_ref = loader(dataset, model_type, False)
model_type = 'resnet501d_good_3042'
test_good = loader(dataset, model_type, False)
model_type = 'resnet50_shuffle_good_3342'
test_shuffle_good = loader(dataset, model_type, False)
model_type = 'resnet501d_bad_1142'
test_bad = loader(dataset, model_type, False)

# print(np.sum(shuffle_good>ref))
import matplotlib.pyplot as plt

# plt.hist(good, bins='auto')  # arguments are passed to np.histogram
# plt.title("Histogram with 'auto' bins")
# plt.show()
# plt.hist(test_ref, bins='auto', alpha=0.5, label='test ref')
plt.hist(test_shuffle_good, bins='auto', alpha=0.5, label='test shuffle good')
plt.hist(test_good, bins='auto', alpha=0.5, label='test 1D good')
plt.legend(loc='upper right')
plt.xlabel('distance')
plt.show()

