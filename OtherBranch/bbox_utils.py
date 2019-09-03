"""
TODO: Before or after running the following scripts, run makedir.sh to ensure you have 1000 folders!!!!!!!!
"""

# TODO: ################################################################################################################
"""
This script selects images from val set each of which has only a single bbox, and the bbox is more or less square.
And tailor the csv files into 2 new csv files which contain only single-bbox image info. But before this, you should
first run process_bounding_boxes.py twice (once with FLOAT=True and once with FLOAT=False) to get:
ILSVRC2012_bbox_val_float.csv
ILSVRC2012_bbox_val_int.csv
"""
# with open('/BS/yfan/nobackup/xmls/ILSVRC2012_bbox_val_int.csv', 'r') as f:
#     txt = f.readlines()
# print('total number of lines:', len(txt))
# name2area = dict()
# name2id = dict()
# for i, line in enumerate(txt):
#     tmp = line[:-1].split(',')
#     # Because image size is preserved at test time, so does the bbox size. So we have to use bbox_val_int
#     xmin = float(tmp[1])
#     ymin = float(tmp[2])
#     xmax = float(tmp[3])
#     ymax = float(tmp[4])
#     if (xmax-xmin) >= (ymax-ymin):
#         ratio = (xmax-xmin) / (ymax-ymin)
#     else:
#         ratio = (ymax - ymin) / (xmax - xmin)
#
#     if ratio >= 1 and ratio <= 1.1:
#         area = (xmax - xmin) * (ymax - ymin)
#         if tmp[0] in name2area.keys():
#             name2area[tmp[0]].append(area)
#             name2id[tmp[0]].append(i)
#         else:
#             name2area[tmp[0]] = [area]
#             name2id[tmp[0]] = [i]
# null_id = []
# for name, area in name2area.items():
#     if len(area) == 1:
#         null_id.extend(name2id[name])
# print(null_id)
# print(len(null_id) / 50000)
#
# with open('/BS/yfan/nobackup/xmls/ILSVRC2012_bbox_val_float.csv', 'r') as f:
#     txt = f.readlines()
#     with open('/BS/yfan/nobackup/xmls/ILSVRC2012_single_bbox_val_float.csv', 'w') as new_f:
#         for i in null_id:
#             new_f.writelines(txt[i])
# with open('/BS/yfan/nobackup/xmls/ILSVRC2012_bbox_val_int.csv', 'r') as f:
#     txt = f.readlines()
#     with open('/BS/yfan/nobackup/xmls/ILSVRC2012_single_bbox_val_int.csv', 'w') as new_f:
#         for i in null_id:
#             new_f.writelines(txt[i])
# TODO: ################################################################################################################
"""
Based on ILSVRC2012_single_bbox_val_int.csv and ILSVRC2012_single_bbox_val_float.csv, split val images into 5 categories
according to their bbox sizes (after resizing to 256). So the folder is like: 
/BS/yfan/nobackup/ILSVRC2012_bboxCrop_val/cropped_8402/bbox??/, where ?? is from 0 to 4. 
Images are first read from '/BS/xian/work/data/imageNet1K/val/' and then cropped based on their bboxes and then put onto
a fix-valued background with the same image sizes and then saved.
This fixed value is the mean value of the dataset, so at test time, those region will have literal no effect to 
the final prediction.
"""
# import os
# import numpy as np
# from PIL import Image
# from tqdm import tqdm
#
# name2class = dict()  # {'ILSVRC2012_val_00000001.JPEG': 'n01751748/' ... 'ILSVRC2012_val_00050000.JPEG': 'n02437616/'}
# with open('/BS/yfan/nobackup/valprep.sh', 'r') as f:
#     txt = f.readlines()
#     for line in txt:
#         if line[1] == 'v':
#             tmp = line[:-2].split(' ')
#             name2class[tmp[1]] = tmp[2]
#
#
# def histedges_equalN(x, nbin):
#     npt = len(x)  # 50000
#     return np.interp(np.linspace(0, npt, nbin + 1),
#                      np.arange(npt),
#                      np.sort(x))
#
#
# def paste_bboxImage_background(src, dst, bbox):
#     # the new image will be filled with the mean value of the dataset, so at test time, those region will have literal
#     # no effect to the final prediction
#     with open(src, 'rb') as f:
#         img = Image.open(f)
#         img.convert('RGB')
#         new_img = Image.new('RGB', img.size, (124, 116, 104))  # mean for test is [0.4863, 0.4549, 0.4078]
#         img = img.crop(bbox)
#         new_img.paste(img, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))
#         new_img.save(dst)
#
#
# def compute_area(xmin_int, ymin_int, xmax_int, ymax_int, xmax_float, ymax_float, ref_size=256):
#     if (xmax_int - xmin_int) >= (ymax_int - ymin_int):
#         ratio = (xmax_int-xmin_int) / (ymax_int-ymin_int)
#     else:
#         ratio = (ymax_int - ymin_int) / (xmax_int - xmin_int)
#     assert ratio >=1 and ratio <= 1.1, 'bbox ratio is out of range of 1.0-1.1!!!'
#     bbox_w = xmax_int - xmin_int
#     bbox_h = ymax_int - ymin_int
#     img_w = xmax_int / xmax_float
#     img_h = ymax_int / ymax_float
#     if (img_w <= img_h and img_w == ref_size) or (img_h <= img_w and img_h == ref_size):
#         scale = 1.0
#     elif img_w < img_h:
#         scale = ref_size / img_w
#     else:
#         scale = ref_size / img_h
#     return bbox_w * scale * bbox_h * scale
#
#
# name2bbox = dict()  # {'ILSVRC2012_val_00000001.JPEG': [0,0,24,157] ... 'ILSVRC2012_val_00050000.JPEG': [23,65,102,510]}
# with open('/BS/yfan/nobackup/xmls/ILSVRC2012_single_bbox_val_int.csv', 'r') as f:
#     txt_int = f.readlines()
#     for i, line in enumerate(txt_int):
#         tmp = line[:-1].split(',')
#         xmin = float(tmp[1])
#         ymin = float(tmp[2])
#         xmax = float(tmp[3])
#         ymax = float(tmp[4])
#         name2bbox[tmp[0]] = [xmin, ymin, xmax, ymax]
#     assert len(name2bbox) == len(txt_int), "name2bbox should be the same length of the csv file!!!"
#
# name2area = dict()  # {'ILSVRC2012_val_00000001.JPEG': 450 ... 'ILSVRC2012_val_00050000.JPEG': 1193}
# area_list = []      # [450, ..., 1193]
# with open('/BS/yfan/nobackup/xmls/ILSVRC2012_single_bbox_val_float.csv', 'r') as f:
#     txt = f.readlines()
#     print('total number of lines:', len(txt))
#     for i, line in enumerate(txt):
#         tmp_float = line[:-1].split(',')
#         tmp_int = txt_int[i][:-1].split(',')
#
#         xmin_float = float(tmp_float[1])
#         ymin_float = float(tmp_float[2])
#         xmax_float = float(tmp_float[3])
#         ymax_float = float(tmp_float[4])
#
#         xmin_int = float(tmp_int[1])
#         ymin_int = float(tmp_int[2])
#         xmax_int = float(tmp_int[3])
#         ymax_int = float(tmp_int[4])
#
#         area = compute_area(xmin_int, ymin_int, xmax_int, ymax_int, xmax_float, ymax_float)
#         if tmp_float[0] in name2area.keys():
#             raise Exception("Some images have multiple bboxes!!!")
#         else:
#             name2area[tmp_float[0]] = area
#             area_list.append(area)
#     assert len(name2area) == len(txt), "name2area should be the same length of the csv file!!!"
#
# num_bins = 5
# bins_borders = histedges_equalN(area_list, num_bins)
# print('the borders of 5 bins are:', bins_borders)
# area_list_bbox = [[] for i in range(num_bins)]
# for name, area in name2area.items():
#     for i in range(num_bins):
#         if area >= bins_borders[i] and area < bins_borders[i+1]:
#             area_list_bbox[i].append(area)
#             break
# for list in area_list_bbox:
#     print('mean is ', np.mean(np.sqrt(list)), 'std is ', np.std(np.sqrt(list)))
#
# src_prefix = '/BS/xian/work/data/imageNet1K/val/'
# dst_prefix = '/BS/yfan/nobackup/ILSVRC2012_bboxCrop_val/cropped_8402_bbox/'
#
# for name, area in tqdm(name2area.items()):
#     if area < bins_borders[1]:
#         src = os.path.join(src_prefix, name)  # /BS/xian/work/data/imageNet1K/val/ILSVRC2012_val_00000001.JPEG
#         if not os.path.isdir(os.path.join(dst_prefix, 'bbox0', name2class[name])):
#             os.makedirs(os.path.join(dst_prefix, 'bbox0', name2class[name]))
#         # /BS/yfan/nobackup/ILSVRC2012_bboxCrop_val/cropped_8402/bbox0/ILSVRC2012_val_00000001.JPEG
#         dst = os.path.join(dst_prefix, 'bbox0', name2class[name], name)
#     elif area < bins_borders[2] and area >= bins_borders[1]:
#         src = os.path.join(src_prefix, name)
#         if not os.path.isdir(os.path.join(dst_prefix, 'bbox1', name2class[name])):
#             os.makedirs(os.path.join(dst_prefix, 'bbox1', name2class[name]))
#         dst = os.path.join(dst_prefix, 'bbox1', name2class[name], name)
#     elif area < bins_borders[3] and area >= bins_borders[2]:
#         src = os.path.join(src_prefix, name)
#         if not os.path.isdir(os.path.join(dst_prefix, 'bbox2', name2class[name])):
#             os.makedirs(os.path.join(dst_prefix, 'bbox2', name2class[name]))
#         dst = os.path.join(dst_prefix, 'bbox2', name2class[name], name)
#     elif area < bins_borders[4] and area >= bins_borders[3]:
#         src = os.path.join(src_prefix, name)
#         if not os.path.isdir(os.path.join(dst_prefix, 'bbox3', name2class[name])):
#             os.makedirs(os.path.join(dst_prefix, 'bbox3', name2class[name]))
#         dst = os.path.join(dst_prefix, 'bbox3', name2class[name], name)
#     else:
#         src = os.path.join(src_prefix, name)
#         if not os.path.isdir(os.path.join(dst_prefix, 'bbox4', name2class[name])):
#             os.makedirs(os.path.join(dst_prefix, 'bbox4', name2class[name]))
#         dst = os.path.join(dst_prefix, 'bbox4', name2class[name], name)
#     paste_bboxImage_background(src, dst, name2bbox[name])
# TODO: ################################################################################################################
"""
Copy images in ILSVRC2012_single_bbox_val_int.csv to /BS/yfan/nobackup/ILSVRC2012_bboxCrop_val/original_8402/
"""
# import os
# from tqdm import tqdm
# from shutil import copyfile
#
# name2class = dict()  # {'ILSVRC2012_val_00000001.JPEG': 'n01751748/' ... 'ILSVRC2012_val_00050000.JPEG': 'n02437616/'}
# with open('/BS/yfan/nobackup/valprep.sh', 'r') as f:
#     txt = f.readlines()
#     for line in txt:
#         if line[1] == 'v':
#             tmp = line[:-2].split(' ')
#             name2class[tmp[1]] = tmp[2]
# name_list = []
# with open('/BS/yfan/nobackup/xmls/ILSVRC2012_single_bbox_val_int.csv', 'r') as f:
#     txt_int = f.readlines()
#     for i, line in enumerate(txt_int):
#         tmp = line[:-1].split(',')
#         name_list.append(tmp[0])
#
# src_prefix = '/BS/xian/work/data/imageNet1K/val/'
# dst_prefix = '/BS/yfan/nobackup/ILSVRC2012_bboxCrop_val/original_8402/'
# for name in tqdm(name_list):
#     src = os.path.join(src_prefix, name)  # /BS/xian/work/data/imageNet1K/val/ILSVRC2012_val_00000001.JPEG
#     if not os.path.isdir(os.path.join(dst_prefix, name2class[name])):
#         os.makedirs(os.path.join(dst_prefix, name2class[name]))
#     # /BS/yfan/nobackup/ILSVRC2012_bboxCrop_val/original_8402/ILSVRC2012_val_00000001.JPEG
#     dst = os.path.join(dst_prefix, name2class[name], name)
#     copyfile(src, dst)
# TODO: ################################################################################################################
"""
Crop all 50000 val images based on the first bbox in the csv file, and then copy it into 
/BS/yfan/nobackup/ILSVRC2012_bboxCrop_val/cropped_50000/
"""
# import os
# from PIL import Image
# from tqdm import tqdm
#
# name2class = dict()  # {'ILSVRC2012_val_00000001.JPEG': 'n01751748/' ... 'ILSVRC2012_val_00050000.JPEG': 'n02437616/'}
# with open('/BS/yfan/nobackup/valprep.sh', 'r') as f:
#     txt = f.readlines()
#     for line in txt:
#         if line[1] == 'v':
#             tmp = line[:-2].split(' ')
#             name2class[tmp[1]] = tmp[2]
#
#
# def paste_bboxImage_background(src, dst, bbox):
#     # the new image will be filled with the mean value of the dataset, so at test time, those region will have literal
#     # no effect to the final prediction
#     with open(src, 'rb') as f:
#         img = Image.open(f)
#         img.convert('RGB')
#         new_img = Image.new('RGB', img.size, (124, 116, 104))  # mean for test is [0.4863, 0.4549, 0.4078]
#         img = img.crop(bbox)
#         new_img.paste(img, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))
#         new_img.save(dst)
#
#
# name2bbox = dict()  # {'ILSVRC2012_val_00000001.JPEG': [0,0,24,157] ... 'ILSVRC2012_val_00050000.JPEG': [23,65,102,510]}
# with open('/BS/yfan/nobackup/xmls/ILSVRC2012_bbox_val_int.csv', 'r') as f:
#     txt_int = f.readlines()
#     for i, line in enumerate(txt_int):
#         tmp = line[:-1].split(',')
#         xmin = float(tmp[1])
#         ymin = float(tmp[2])
#         xmax = float(tmp[3])
#         ymax = float(tmp[4])
#         if tmp[0] in name2bbox.keys():
#             continue
#         else:
#             name2bbox[tmp[0]] = [xmin, ymin, xmax, ymax]
#     assert len(name2bbox) == 50000, "name2bbox should be 50000!!!"
#
# src_prefix = '/BS/xian/work/data/imageNet1K/val/'
# dst_prefix = '/BS/yfan/nobackup/ILSVRC2012_bboxCrop_val/cropped_50000/'
# for name, bbox in tqdm(name2bbox.items()):
#     src = os.path.join(src_prefix, name)  # /BS/xian/work/data/imageNet1K/val/ILSVRC2012_val_00000001.JPEG
#     if not os.path.isdir(os.path.join(dst_prefix, name2class[name])):
#         os.makedirs(os.path.join(dst_prefix, name2class[name]))
#     # /BS/yfan/nobackup/ILSVRC2012_bboxCrop_val/cropped_50000/ILSVRC2012_val_00000001.JPEG
#     dst = os.path.join(dst_prefix, name2class[name], name)
#     paste_bboxImage_background(src, dst, bbox)