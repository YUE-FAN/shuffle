"""
Use bbox.csv to re-arrange the ILSVRC2012/val/ into ILSVRC2012/mean/bbox???/,
where ??? is the size of the bbox. So inside each bbox???/, we still have 1000 class,
just the number of val image for each class is different. We will use histogram to determine 10
different image areas and then split images accordingly.

You have to first run /BS/yfan/nobackup/mkdir.sh to create the structure of the folders for copying images
"""
import os
import numpy as np
from shutil import copyfile
from tqdm import tqdm

name2class = dict()  # {'ILSVRC2012_val_00000001.JPEG': 'n01751748/' ... 'ILSVRC2012_val_00050000.JPEG': 'n02437616/'}
id2name = dict()
with open('/BS/yfan/nobackup/valprep.sh', 'r') as f:
    txt = f.readlines()
k = 0
for line in txt:
    if line[1] == 'v':
        tmp = line[:-2].split(' ')
        name2class[tmp[1]] = tmp[2]
        id2name[k] = tmp[1]
        k += 1


def histedges_equalN(x, nbin):
    npt = len(x)  # 50000
    return np.interp(np.linspace(0, npt, nbin + 1),
                     np.arange(npt),
                     np.sort(x))


with open('/BS/yfan/nobackup/xmls/ILSVRC2012_bbox_val.csv', 'r') as f:
    txt = f.readlines()
print('total number of lines:', len(txt))
name2area = dict()
for i, line in enumerate(txt):
    tmp = line[:-1].split(',')
    xmin = float(tmp[1])
    ymin = float(tmp[2])
    xmax = float(tmp[3])
    ymax = float(tmp[4])
    area = (xmax-xmin)*(ymax-ymin)
    if tmp[0] in name2area.keys():
        name2area[tmp[0]].append(area)
    else:
        name2area[tmp[0]] = [area]

null_list = []    # get rid of multilabeled images
mean_list = []    # take the mean of multilabeled images
median_list = []  # take the median of multilabeled images
max_list = []     # take the max of multilabeled images
for name, area in name2area.items():
    if len(area) == 1:
        null_list.append(area[0])
    mean_list.append(np.mean(area))
    max_list.append(np.max(area))
    median_list.append(np.median(area))

num_bins = 5
mean_bins = histedges_equalN(mean_list, num_bins)
max_bins = histedges_equalN(max_list, num_bins)
median_bins = histedges_equalN(median_list, num_bins)
null_bins = histedges_equalN(null_list, num_bins)

for i in tqdm(range(50000)):
    area = mean_list[i]
    if area < mean_bins[1]:
        src = os.path.join('/BS/xian/work/data/imageNet1K/val/', id2name[i])
        dst = os.path.join('/BS/yfan/nobackup/ILSVRC2012_bbox_val/mean/bbox0', name2class[id2name[i]], id2name[i])
    elif area < mean_bins[2] and area >= mean_bins[1]:
        src = os.path.join('/BS/xian/work/data/imageNet1K/val/', id2name[i])
        dst = os.path.join('/BS/yfan/nobackup/ILSVRC2012_bbox_val/mean/bbox1', name2class[id2name[i]], id2name[i])
    elif area < mean_bins[3] and area >= mean_bins[2]:
        src = os.path.join('/BS/xian/work/data/imageNet1K/val/', id2name[i])
        dst = os.path.join('/BS/yfan/nobackup/ILSVRC2012_bbox_val/mean/bbox2', name2class[id2name[i]], id2name[i])
    elif area < mean_bins[4] and area >= mean_bins[3]:
        src = os.path.join('/BS/xian/work/data/imageNet1K/val/', id2name[i])
        dst = os.path.join('/BS/yfan/nobackup/ILSVRC2012_bbox_val/mean/bbox3', name2class[id2name[i]], id2name[i])
    else:
        src = os.path.join('/BS/xian/work/data/imageNet1K/val/', id2name[i])
        dst = os.path.join('/BS/yfan/nobackup/ILSVRC2012_bbox_val/mean/bbox4', name2class[id2name[i]], id2name[i])
    copyfile(src, dst)

for i in tqdm(range(50000)):
    area = max_list[i]
    if area < max_bins[1]:
        src = os.path.join('/BS/xian/work/data/imageNet1K/val/', id2name[i])
        dst = os.path.join('/BS/yfan/nobackup/ILSVRC2012_bbox_val/max/bbox0', name2class[id2name[i]], id2name[i])
    elif area < max_bins[2] and area >= max_bins[1]:
        src = os.path.join('/BS/xian/work/data/imageNet1K/val/', id2name[i])
        dst = os.path.join('/BS/yfan/nobackup/ILSVRC2012_bbox_val/max/bbox1', name2class[id2name[i]], id2name[i])
    elif area < max_bins[3] and area >= max_bins[2]:
        src = os.path.join('/BS/xian/work/data/imageNet1K/val/', id2name[i])
        dst = os.path.join('/BS/yfan/nobackup/ILSVRC2012_bbox_val/max/bbox2', name2class[id2name[i]], id2name[i])
    elif area < max_bins[4] and area >= max_bins[3]:
        src = os.path.join('/BS/xian/work/data/imageNet1K/val/', id2name[i])
        dst = os.path.join('/BS/yfan/nobackup/ILSVRC2012_bbox_val/max/bbox3', name2class[id2name[i]], id2name[i])
    else:
        src = os.path.join('/BS/xian/work/data/imageNet1K/val/', id2name[i])
        dst = os.path.join('/BS/yfan/nobackup/ILSVRC2012_bbox_val/max/bbox4', name2class[id2name[i]], id2name[i])
    copyfile(src, dst)

for i in tqdm(range(50000)):
    area = median_list[i]
    if area < median_bins[1]:
        src = os.path.join('/BS/xian/work/data/imageNet1K/val/', id2name[i])
        dst = os.path.join('/BS/yfan/nobackup/ILSVRC2012_bbox_val/median/bbox0', name2class[id2name[i]], id2name[i])
    elif area < median_bins[2] and area >= median_bins[1]:
        src = os.path.join('/BS/xian/work/data/imageNet1K/val/', id2name[i])
        dst = os.path.join('/BS/yfan/nobackup/ILSVRC2012_bbox_val/median/bbox1', name2class[id2name[i]], id2name[i])
    elif area < median_bins[3] and area >= median_bins[2]:
        src = os.path.join('/BS/xian/work/data/imageNet1K/val/', id2name[i])
        dst = os.path.join('/BS/yfan/nobackup/ILSVRC2012_bbox_val/median/bbox2', name2class[id2name[i]], id2name[i])
    elif area < median_bins[4] and area >= median_bins[3]:
        src = os.path.join('/BS/xian/work/data/imageNet1K/val/', id2name[i])
        dst = os.path.join('/BS/yfan/nobackup/ILSVRC2012_bbox_val/median/bbox3', name2class[id2name[i]], id2name[i])
    else:
        src = os.path.join('/BS/xian/work/data/imageNet1K/val/', id2name[i])
        dst = os.path.join('/BS/yfan/nobackup/ILSVRC2012_bbox_val/median/bbox4', name2class[id2name[i]], id2name[i])
    copyfile(src, dst)

for i in tqdm(range(len(null_list))):
    area = null_list[i]
    if area < null_bins[1]:
        src = os.path.join('/BS/xian/work/data/imageNet1K/val/', id2name[i])
        dst = os.path.join('/BS/yfan/nobackup/ILSVRC2012_bbox_val/null/bbox0', name2class[id2name[i]], id2name[i])
    elif area < null_bins[2] and area >= null_bins[1]:
        src = os.path.join('/BS/xian/work/data/imageNet1K/val/', id2name[i])
        dst = os.path.join('/BS/yfan/nobackup/ILSVRC2012_bbox_val/null/bbox1', name2class[id2name[i]], id2name[i])
    elif area < null_bins[3] and area >= null_bins[2]:
        src = os.path.join('/BS/xian/work/data/imageNet1K/val/', id2name[i])
        dst = os.path.join('/BS/yfan/nobackup/ILSVRC2012_bbox_val/null/bbox2', name2class[id2name[i]], id2name[i])
    elif area < null_bins[4] and area >= null_bins[3]:
        src = os.path.join('/BS/xian/work/data/imageNet1K/val/', id2name[i])
        dst = os.path.join('/BS/yfan/nobackup/ILSVRC2012_bbox_val/null/bbox3', name2class[id2name[i]], id2name[i])
    else:
        src = os.path.join('/BS/xian/work/data/imageNet1K/val/', id2name[i])
        dst = os.path.join('/BS/yfan/nobackup/ILSVRC2012_bbox_val/null/bbox4', name2class[id2name[i]], id2name[i])
    copyfile(src, dst)


