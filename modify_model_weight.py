import torch
import os
import shutil


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    if not os.path.isdir(checkpoint):
        os.makedirs(checkpoint)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


# Load checkpoint.
checkpoint_path = "/data/users/yuefan/fanyue/dconv/checkpoints/stl10/dconv_none_vgg16_42_winu_60/model_best.pth.tar"
print('==> Resuming from checkpoint..')
assert os.path.isfile(checkpoint_path), 'Error: no checkpoint directory found!'

checkpoint = torch.load(checkpoint_path)
best_acc = checkpoint['best_acc']
start_epoch = checkpoint['epoch']
a = checkpoint['state_dict']['module.conv52.conv.weight']
print(type(a))
print(a.size())

# # randomly deletion based on units
# a = a.view(-1)
# perm = torch.randperm(512*512*9)
# perm = perm[0:106255*9]
# a[perm] = 0

# # delete the least important ones based on pages
# for i in range(512):
#     for j in range(512):
#         c = a[i, j, :, :].view(-1)
#         b = torch.sum(torch.abs(c))  # the summed map 512 x 1
#         if b >= 0.16:
#             a[i, j, :, :] = 0

# # delete the least important ones based on units
# a = a.view(-1)
# _, b = torch.sort(torch.abs(a), dim=0, descending=False)
# a[b[0:130530*9]] = 0
#
# print(130530/512/512)

# randomly deletion based on pages
a = a.view(512*512, 9)
perm = torch.randperm(512*512)
perm = perm[0:130530]
a[perm, :] = 0


save_path = "/data/users/yuefan/fanyue/dconv/checkpoints/stl10/dconv_none_vgg16_42_winu_60_mod/model_best.pth.tar"
save_checkpoint(checkpoint, True, checkpoint=os.path.dirname(save_path))
print(checkpoint['state_dict']['module.conv52.conv.weight'].size())
# for key, value in enumerate(checkpoint['state_dict'].items()):
#     print(value[0])
# model.load_state_dict(checkpoint['state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer'])
# logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
