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