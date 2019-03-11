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
# a = np.load('/nethome/yuefan/fanyue/dconv/x.npy')
# print(a.shape)
# a = a.reshape(128, 512, 4, 4)
# # print(np.sum(a[4,:,:], 1))
# a = a.reshape(-1)
#
# b = []
# for i in range(len(a)):
#     if a[i] != 0:
#         b.append(a[i])
# print(len(b) / len(a))
#
# import matplotlib.pyplot as plt
#
# plt.hist(b, bins='auto')  # arguments are passed to np.histogram
# plt.title("Histogram with 'auto' bins")
# plt.show()

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

import numpy as np
import matplotlib.pyplot as plt

x = range(13)
r100 = np.ones(shape=(13,)) * 30.82
r10 = np.ones(shape=(13,)) * 74.47
a10 =[

10.00 ,
10.00 ,
10.00 ,
10.00 ,
10.00 ,
48.45 ,
78.77 ,
72.59 ,
76.85 ,
77.20 ,
74.32 ,
73.49 ,
75.84


]
a100=[

1.00 ,
1.00 ,
1.00 ,
1.00 ,
1.00 ,
1.00 ,
4.15 ,
28.82 ,
21.40 ,
31.46 ,
28.91 ,
30.08 ,
29.53
]

plt.plot(x,r100, label="CIFAR100 stand")
plt.plot(x,a100, label="CIFAR100 1D")
plt.plot(x,r10, label="CIFAR10 stand")
plt.plot(x,a10, label="CIFAR10 1D")
plt.legend(loc='lower right')
plt.title("VGG16 1D w/o BN")
plt.xlabel("number of standard layers")
plt.ylabel("test acc")

plt.show()
