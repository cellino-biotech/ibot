
import argparse
import os
import sys
import datetime
import time
import math
import json
import glob
import numpy as np
import utils
import models
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from pathlib import Path
from PIL import Image
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
from tensorboardX import SummaryWriter
from models.head import iBOTHead
from loader import ImageFolderMask
# from evaluation.unsupervised.unsup_cls import eval_pred


from cellino_tfdata_loader import TFRecords, density_loader, file_list, WB_DIR, Zslice, Mask

data_path = "/home/slee_cellinobio_com/dl-data1/data/TFRECORD-4x_density/rechunk/train*.tfr"
file_list = glob.glob(data_path)


train_tfr_list = sorted(file_list)[:6]
nslice = 2
image_name_list = [str(i) for i in range(nslice)]
fcn_list = [Zslice(0, 4, 2, nslice), Mask(16, 0.3, 0, (0.3, 1/0.3), image_name_list, 'block')]

ds = TFRecords(density_loader, train_tfr_list, nsample_per_file=100, process_fcn_list=fcn_list)

for i, data in enumerate(iter(ds)):
    print(i)
  
subdata = density_original_loader(train_tfr_list).load_data().as_numpy_iterator()

xx = next(iter(subdata))



#%%
from torchdata.datapipes.iter import FileOpener



# list_cpu = tf.config.list_physical_devices('CPU')
# with tf.device('/device:CPU:0'):
#     for i, data in enumerate(iter(ds)):
#         print(i)

# data_loader = torch.utils.data.DataLoader(ds, batch_size=24, pin_memory=True)

# for i, batch in enumerate(data_loader):
#     print(i)
#     # print(batch.keys())
#     print(batch['msk_0'].shape, batch['msk_1'].shape)



# #%% original ibot/dino data loader testing
# import torch
# from main_ibot import DataAugmentationiBOT
# from loader import ImageFolderMask
# class args:
#     global_crops_scale = (0.4, 1.)
#     global_crops_number = 2
#     local_crops_number = 0
#     local_crops_scale = (0.05, 0.4)
#     data_path = '/media/slee/DATA1/DL/data/tiny-imagenet-200/train'
#     pred_ratio = 0.3
#     pred_ratio_var = 0
#     pred_shape = 'block'
#     pred_start_epoch = 0
#     num_workers = 5
#     batch_size_per_gpu = 10


# transform = DataAugmentationiBOT(
#         args.global_crops_scale,
#         args.local_crops_scale,
#         args.global_crops_number,
#         args.local_crops_number,
#     )
# pred_size = 16
# dataset = ImageFolderMask(
#     args.data_path, 
#     transform=transform,
#     patch_size=pred_size,
#     pred_ratio=args.pred_ratio,
#     pred_ratio_var=args.pred_ratio_var,
#     pred_aspect_ratio=(0.3, 1/0.3),
#     pred_shape=args.pred_shape,
#     pred_start_epoch=args.pred_start_epoch)
# # sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
# data_loader = torch.utils.data.DataLoader(
#     dataset,
#     # sampler=sampler,
#     batch_size=args.batch_size_per_gpu,
#     num_workers=args.num_workers,
#     pin_memory=True,
#     drop_last=True
# )

# for batch_data in data_loader:
#     pass

# image, label, mask = batch_data



# image, label, mask = dataset[0]