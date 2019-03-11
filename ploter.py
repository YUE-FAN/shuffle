"""
This code analyzes the Linear Model trained on MNIST which achieves 92.69% test acc and 93.27% train acc
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
w = np.load('/nethome/yuefan/fanyue/dconv/mnist/l_w.npy')
num = 9
img = np.load('/nethome/yuefan/fanyue/dconv/mnist/img'+str(num)+'.npy')
print(np.max(img), np.min(img))
# import matplotlib.pyplot as plt
# plt.hist(w.reshape(-1,), bins='auto')
# plt.show()
# for i in range(10):
#     y = np.sum(w[i, 0, :, :] * img)
#     print(y)
w0 = w[num, 0, :, :].reshape(28, 28)
for i in range(28):
    for j in range(28):
        if w0[i,j]>-0.15 and w0[i,j]<0.15:
            w0[i,j] = 0
print(np.max(w0), np.min(w0))
#
w0 = w0 - np.min(w0)
w0 = w0 / np.max(w0)
# print(w0)
print(np.max(w0), np.min(w0))
plt.figure(1)
plt.subplot(211)
plt.imshow(img, cmap='hot', interpolation='nearest')
plt.subplot(212)
plt.imshow(w0, cmap='hot', interpolation='nearest')
plt.show()
# Image.fromarray(np.hstack(( np.uint8(w0*255), np.uint8(img*255) ))).show()
# im = Image.fromarray(np.uint8(w0*255))
# im.show()
# TODO:################################################################################################################
"""
This code can plot the histogram of cos similarities among all activations on the last feature map
I sampled 100 images
"""
# import numpy as np
# a = np.load('/nethome/yuefan/fanyue/dconv/dist_lists.npy')
#
# a = a.transpose(1, 0, 2)
# print(a.shape)
# iu1 = np.triu_indices(64)
# ll = []
# for i in range(100):
#     b = a[i]
#     c = b[iu1]
#     ll.extend(list(c))
#
# import matplotlib.pyplot as plt
#
# plt.hist(ll, bins='auto')  # arguments are passed to np.histogram
# plt.title("Histogram with 'auto' bins")
# plt.show()
# TODO:################################################################################################################
"""
This code can plot the histogram of the magnitude of the maps for a certain layer.
I expected an exponential distribution.
"""
# import numpy as np
# number = 52
# a = np.load('/nethome/yuefan/fanyue/dconv/weight' + str(number) + '.npy')
# print(a.shape)
"""
This part checks if shuffle or rand can make the weight values on the same page the same, because anyway the spatial
relation doesn't matter, so I guess this will drive the weight values on the same page the same, but different pages
in one kernel should have different weight values, for example, kernel0:
[[0.1,0.1,0.1]  [9,9,9]
 [0.1,0.1,0.1]  [9,9,9]
 [0.1,0.1,0.1]  [9,9,9]]
"""
# var = []
# abssum = []
# for j in range(a.shape[0]):
#     c = a[j, :, :, :].reshape(a.shape[1], a.shape[2]*a.shape[3])
#     for i in range(a.shape[1]):
#         var.append(np.var(c[i]))
#         abssum.append(np.sum(abs(c[i])))
# import matplotlib.pyplot as plt
# plt.scatter(abssum, np.log(var))
# plt.xlabel('L1 norm')
# plt.ylabel('log var of the activation values at a page')
# plt.title('shuffle_vgg16_53')
# plt.show()

"""
This part plots the L1 norm of the feature map pages
"""
# c = a.reshape(a.shape[0], a.shape[1], a.shape[2] * a.shape[3])
# b = np.sum(abs(c), axis=2)
# b = b.reshape(a.shape[0], a.shape[1])  # the summed map 512^2 x 1
# for i in range(a.shape[0]):
#     d = b[i]
#     import matplotlib.pyplot as plt
#     plt.hist(d, bins='auto')  # arguments are passed to np.histogram
#     plt.title("Histogram of feature map i before none_vgg16_53")
#     plt.show()

# for i in range(a.shape[0]):
#     d = a[i]
# import matplotlib.pyplot as plt
# plt.hist(a.reshape(-1), bins='auto')  # arguments are passed to np.histogram
# plt.title("Histogram of fc weights of none_vgg16_52")
# plt.show()


"""
This part answers how many channels are important for a kernel? 
kk is the importance threshold. For each kernel, d takes all the indices of the channels whose sum exceed the threshold.
Thus the length of d is the number of channels that are important for this kernel.
p stores the number of channels that are important for all kernels. So, the mean of p is "averagely speaking, how many 
channels are important for one kernel given a threshold kk"

Note that if you plot the hist of indices of channels that are important for all kernels,
you see an uniform distribution indicating all channels are useful, either to you or to me.
"""
# tt = []
# for k in range(100):
#     kk = k/100
#     p = []
#     for j in range(512):
#         c = a[j, :, :, :]
#         c = c.reshape(512, 9)
#         b = np.sum(c, axis=1)
#     # print(min(abs(b)))
#     # b = b.reshape(512*512)
#         d = []
#         for i in range(512):
#             if abs(b[i]) >= kk:
#                 d.append(i)
#                 # print(b[i])
#                 # print(c[i, :])
#                 # print('==============================')
#         # print(len(d))
#         p.append(len(d))
#     tt.append(sum(p)/len(p))
#
# # print(max(d), min(d), len(d))
# import matplotlib.pyplot as plt
# plt.plot(np.arange(0,1,0.01), tt)
# print(tt)
# plt.show()
"""
This part sum over all kernels
"""
# c = a.reshape(512, 512, 9)
# b = np.sum(c, axis=2)
# b = b.reshape(512*512)  # the summed map 512^2 x 1
# import matplotlib.pyplot as plt
# plt.hist(b, bins='auto')  # arguments are passed to np.histogram
# # print()
# plt.title("Histogram with 'auto' bins")
# plt.show()
"""
This part sum over the abs of all kernels
"""
# c = a.reshape(a.shape[0], a.shape[1] * 9)
# b = np.sum(abs(c), axis=1)
# b = b.reshape(a.shape[0])  # the summed map 512^2 x 1
# import matplotlib.pyplot as plt
# plt.hist(b, bins='auto')  # arguments are passed to np.histogram
# plt.title("Histogram with 'auto' bins")
# plt.show()
"""
This part sum over one kernels
"""
# for i in range(a.shape[0]):
#     c = a[i, :, :, :]
#     c = c.reshape(a.shape[1], 9)
#     b = np.sum(abs(c), axis=1)  # the summed map 512 x 1
#     # print(max(d), min(d), len(d))
#     import matplotlib.pyplot as plt
#     plt.hist(b, bins='auto')  # arguments are passed to np.histogram
#     plt.title("Histogram with 'auto' bins")
#     plt.show()


# for i in range(512):
#     for j in range(512):
#         c = a[i, j, :, :].reshape(-1)
#         b = np.sum(abs(c))  # the summed map 512 x 1
#         if b <= 0.06:
#             a[i, j, :, :] = 0

"""
This part computes how many pages should be dropped based on L1 norm for modify_model_weight.py
"""
# c = a.reshape(a.shape[0], a.shape[1], 9)
# b = np.sum(abs(c), axis=2)
# b = b.reshape(a.shape[0] * a.shape[1])  # the summed map 512^2 x 1
# d = []
# for i in range(a.shape[0] * a.shape[1]):
#     if b[i] <= 0.15:
#         d.append(b[i])
# print("this many pages should be dropped", len(d))
# print("this percent pages should be dropped", len(d)/512/512)
# import matplotlib.pyplot as plt
# plt.hist(b, bins='auto')  # arguments are passed to np.histogram
# plt.title("Histogram with 'auto' bins")
# plt.show()
"""
This part computes the cos similarity of the pages of a kernel
"""
# from sklearn.metrics.pairwise import cosine_similarity
# import matplotlib.pyplot as plt
# for i in range(a.shape[0]):
#     c = a[i, :, :, :]
#     c = c.reshape(a.shape[1], 9)
#     sim = cosine_similarity(c)
#
#     plt.hist(sim.reshape(-1), bins='auto')  # arguments are passed to np.histogram
#     plt.title("Histogram with 'auto' bins")
#     plt.show()
"""
This part computes the cos similarity of kernels
"""
# from sklearn.metrics.pairwise import cosine_similarity
# import matplotlib.pyplot as plt
# c = a.reshape(a.shape[0], 9*a.shape[1])
# sim = cosine_similarity(c)
#
# plt.hist(sim.reshape(-1), bins='auto')  # arguments are passed to np.histogram
# plt.title("Histogram of the cos similarity of kernels at layer " + str(number))
# plt.xlabel('cos similarity')
# plt.show()
