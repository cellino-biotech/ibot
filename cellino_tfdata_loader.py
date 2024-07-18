import os
import math
import random
import glob
import numpy as np
import pandas as pd
import copy
from functools import *
import timeit
import time
import wandb

import argparse
import tensorflow as tf
import torch
import torch.distributed as dist
import torch.nn as nn


# from cellino_utils.density_data2d import DensityData2D
from helpers import cellinoTFRreader


WB_DIR = 'mnt/disks/dl-data1/wandb'
# WB_DIR = '/Users/sangkyunlee/Cellino/DL/data/wandb'
# WB_DIR = '/media/slee/DATA1/DL/data/wandb'
wb_entity = 'cellino-ml-ninjas'
wb_project_name = 'WP_CELL_ID-4x_density'
artifact_name = 'TFRECORD-4x_density'
ver = 'latest'

# api = wandb.Api()
# artifact = api.artifact(f'{wb_entity}/{wb_project_name}/{artifact_name}:{ver}')#, type='dataset_with_coarse_clean')
# art_dir = artifact.download(root=os.path.join(WB_DIR, artifact_name))

art_dir = os.path.join(WB_DIR, artifact_name)
file_tables = glob.glob(os.path.join(art_dir, "*.csv"))

if os.path.exists(file_tables[0]):
    data_info = pd.read_csv(file_tables[0])


file_list = glob.glob(os.path.join(art_dir, '*-EDGE.tfr'))




train_tfr_list =sorted(file_list)[:10]
nchannel_in_data = 4
patch_shape = (256, 256)
dataready = False
#dummy values
z_indices = None #[0]
shuffle_buffer_size = None #256
drop_remainder = None #False
train_batch_size = None #10
multiscale_list = [] #[0]
density_scale = None
cutoff_overgrown = None
cutoff_empty = None
nrepeat = None

DEN_FEAT_SHAPE = [(nchannel_in_data,) + patch_shape, (1,) + patch_shape, (1,)]
DEN_FEAT_NAME = ['brt', 'density', 'patch_index']
DEN_FEAT_DATA_TYPE = [tf.float32, tf.float32, tf.int64]



# density_loader  = partial(DensityData2D,
#                           nchannel_in_data=nchannel_in_data,
#                         patch_shape=patch_shape,
#                         dataready=dataready,
#                         multiscale_list=multiscale_list,
#                         density_scale=density_scale,
#                         cutoff_overgrown=cutoff_overgrown,
#                         cutoff_empty=cutoff_empty,
#                         nrepeat=nrepeat,
#                         zindices=z_indices,
#                         shuffle_buffer_size=shuffle_buffer_size,
#                         drop_remainder=drop_remainder,
#                         batch_size=train_batch_size)



density_original_loader = partial(cellinoTFRreader,
                                  features_shape=DEN_FEAT_SHAPE,
                                  features_name=DEN_FEAT_NAME,
                                  features_data_type=DEN_FEAT_DATA_TYPE)
                                  

new_density_loader = partial(cellinoTFRreader,
                              features_shape=DEN_FEAT_SHAPE + [None],
                                  features_name=DEN_FEAT_NAME +['tfr_name'],
                                  features_data_type=DEN_FEAT_DATA_TYPE + [tf.string])

def _to_feature_bytes(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _to_feature_float(arr):
    if isinstance(arr, (float, np.ndarray)):
        arr = np.array(arr)
    return tf.train.Feature(float_list=tf.train.FloatList(value=arr.flatten()))

def _to_feature_int(arr):
    if isinstance(arr, (int, np.ndarray)):
        arr = np.array(arr)
    return tf.train.Feature(int64_list=tf.train.Int64List(value=arr.flatten()))

def _convert_tf_feature(array):
    def _check_int(dtype):
        boolean = (dtype == np.int8) or (dtype == np.int16) or (dtype == np.int32) or (dtype == np.int64)\
                or (dtype == np.uint8) or (dtype == np.uint16) or (dtype == np.uint32) or (dtype == np.uint64)
        return boolean
    if (array.dtype == np.float32):
        return _to_feature_float(array.flatten().astype('float32'))
    elif _check_int(array.dtype):
        return _to_feature_int(array.flatten())
    else:
        raise ValueError("Not permitted data type is given")
        # logger.debug("Not permitted data type is given")

def convert_np2tfr(numpy_data_dict):
    out = dict()

    for key, val in numpy_data_dict.items():
        assert isinstance(val, np.ndarray), f"{key} should be a numpy array."
        out[key] = _convert_tf_feature(val)
    return out


def rechunk_tfrecords(tfr_loader, newtfr_name, nchunk=10, write_tfrinfo=False):
    dataset = tfr_loader.load_data().as_numpy_iterator()
    tfr_list = getattr(tfr_loader, 'tfr_name_list') if hasattr(tfr_loader, 'tfr_name_list') else []

    tfwriter = None
    tfr_index = -1
    pre_patch_index = 0
    for i, np_data in enumerate(dataset):
        if i % nchunk == 0:
            chunk_start = True
        else:
            chunk_start = False
        
        chunk_no = i // nchunk
        if chunk_start:
            newtfr_chunk = newtfr_name + f'_{chunk_no}.tfr'
            tfwriter = tf.io.TFRecordWriter(newtfr_chunk)

        feats = convert_np2tfr(np_data)
        if write_tfrinfo and\
            isinstance(np_data, dict) and\
            'patch_index' in np_data.keys():

            patch_index = np_data['patch_index']
            if patch_index == 0:
                tfr_index += 1
            else:
                assert patch_index > pre_patch_index,\
                    "ONLY ALLOWED IF patch_index is incremental within a TFR and start from 0."
            fn = tfr_list[tfr_index].split('/')[-1]
            feats['tfr_name'] = _to_feature_bytes(fn.encode('utf-8'))
            pre_patch_index = patch_index

        features = tf.train.Features(feature=feats)
        record_bytes = tf.train.Example(features=features).SerializeToString()
        tfwriter.write(record_bytes)
        if i % nchunk == nchunk-1 and tfwriter:
            tfwriter.close()
            tfwriter = None
    
    if tfwriter:
        tfwriter.close()

    last_chunk_sample = (i % nchunk) + 1
    return chunk_no+1, last_chunk_sample

class TFRecords(torch.utils.data.IterableDataset):
    """To read cellino TFRecords list. 
    This class generates sub-dataset for multi-gpu training and thus 
    No need to use torch.utils.data.distributed.DistributedSampler
    NOTE: all the TFRecords should have the same number of samples.

    """
    def __init__(self, loader, list_tfrs,  nsample_per_file, process_fcn_list=[], shuffle_buffer_size=512, feat_codes={'X': 'brt', 'Y': 'density'}):
        super(TFRecords).__init__()

        overall_start = 0
        overall_end = len(list_tfrs)
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            worker_id = int(os.environ['RANK'])
            num_workers = int(os.environ['WORLD_SIZE'])
        else:
            worker_id = 0
            num_workers = 1
            print("Failed to load world_size and/or rank from os.environ and set a single data loader ")
        
        per_worker = int(math.ceil((overall_end - overall_start) / float(num_workers)))
        start = overall_start + worker_id * per_worker
        end = min(start + per_worker, overall_end)
        assert end > start, "this example code only works with end >= start"
        self.start = start
        self.end = end
        assert end <= len(list_tfrs)
        self.feat_codes = feat_codes
        self.list_files = list_tfrs
        self.loader = loader
        self.shuffle_buffer_size = shuffle_buffer_size
        self.nsample = (end - start) * nsample_per_file
        self.process_fcn_list = process_fcn_list
        # self.sample_count = 0
    def __iter__(self):

        subdata = self.loader(self.list_files[self.start:self.end]).load_data()
        if len(self.process_fcn_list) > 0:
            for fcn in self.process_fcn_list:
                subdata = subdata.map(fcn)
        if self.shuffle_buffer_size > 0:
            subdata = subdata.shuffle(self.shuffle_buffer_size)
        return subdata.as_numpy_iterator()

    def __len__(self):
        return self.nsample

class Zslice(object):
    def __init__(self, istart, iend, nsample, min_slice_gap, max_slice_gap=None, channel_first=True, img_code={'X':'brt'}):
        self.istart = istart
        self.iend = iend
        self.min_slice_gap = min_slice_gap
        if not max_slice_gap:
            self.max_slice_gap = min_slice_gap
        else:
            self.max_slice_gap = max_slice_gap

        max_indices = np.arange(self.istart, self.iend, self.min_slice_gap)
        assert len(max_indices) <= nsample, f"we can only sample {len(max_indices)}." 
        self.nsample = nsample

        self.channel_first = channel_first
        self.img_code = img_code

    def __call__(self, data):

        img_stack = data[self.img_code['X']]
        assert len(tf.shape(img_stack)) == 3, "image stack should be in 3D."
        if not self.channel_first:
            img_stack = tf.transpose(img_stack, (2, 0, 1))
        
        nslice = len(img_stack)
        assert nslice <= (self.iend - self.istart), "search range error."


        indices = []
        i = 0
        while len(indices) < self.nsample:
            slice_step = int(random.uniform(self.min_slice_gap, self.max_slice_gap))

            istart = int(random.uniform(self.istart, self.iend - slice_step - 1))
            iend = min(self.iend, nslice)
            indices = np.arange(istart, iend, slice_step)
            i += 1
            if i>100:
                print("failed to load proper indices!")
                break


        # NOTE: TF cannot handle a list of different sizes of tensors 
        # probably because we later convert the dataset as numpy iterator.
        # Thus, added a new transformed stack into the data dictionary
        # with the key of indice
        # x = int(random.uniform(100, 200))
        # y = int(random.uniform(100, 200))
        # out_stack = [tf.slice(img_stack, begin=[i,0,0], size=[1, x, y]) for i in indices[:self.nsample]]

        for i, ind in enumerate(indices[:self.nsample]):
            x = 128#int(random.uniform(100, 200))
            y = 128#int(random.uniform(100, 200))
            data[f'{i}'] = tf.slice(img_stack, begin=[ind, 0, 0], size=[1, x, y])
        # data['zslice'] = out_stack
        return data




class Mask:
    """ Adaptation of iBOT's ImageFolderMask class to load Cellino TFRecords"""
    def __init__(self, patch_size, pred_ratio, pred_ratio_var, pred_aspect_ratio, 
                 image_name_list,
                 pred_shape='block', pred_start_epoch=0):
        
        self.psz = patch_size
        self.pred_ratio = pred_ratio[0] if isinstance(pred_ratio, list) and \
            len(pred_ratio) == 1 else pred_ratio
        self.pred_ratio_var = pred_ratio_var[0] if isinstance(pred_ratio_var, list) and \
            len(pred_ratio_var) == 1 else pred_ratio_var
        if isinstance(self.pred_ratio, list) and not isinstance(self.pred_ratio_var, list):
            self.pred_ratio_var = [self.pred_ratio_var] * len(self.pred_ratio)
        self.log_aspect_ratio = tuple(map(lambda x: math.log(x), pred_aspect_ratio))
        self.pred_shape = pred_shape
        self.pred_start_epoch = pred_start_epoch
        self.image_name_list = image_name_list

    def get_pred_ratio(self):
        if hasattr(self, 'epoch') and self.epoch < self.pred_start_epoch:
            return 0

        if isinstance(self.pred_ratio, list):
            pred_ratio = []
            for prm, prv in zip(self.pred_ratio, self.pred_ratio_var):
                assert prm >= prv
                pr = random.uniform(prm - prv, prm + prv) if prv > 0 else prm
                pred_ratio.append(pr)
            pred_ratio = random.choice(pred_ratio)
        else:
            assert self.pred_ratio >= self.pred_ratio_var
            pred_ratio = random.uniform(self.pred_ratio - self.pred_ratio_var, self.pred_ratio + \
                self.pred_ratio_var) if self.pred_ratio_var > 0 else self.pred_ratio
        
        return pred_ratio

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __call__(self, data):
        print(self.image_name_list)

        for ikey in self.image_name_list:
            print('data_key: ', ikey)
            img = data[ikey]
            H, W = img.shape[1] // self.psz, img.shape[2] // self.psz

            high = self.get_pred_ratio() * H * W


   
            if self.pred_shape == 'block':
                # following BEiT (https://arxiv.org/abs/2106.08254), see at
                # https://github.com/microsoft/unilm/blob/b94ec76c36f02fb2b0bf0dcb0b8554a2185173cd/beit/masking_generator.py#L55
                mask = np.zeros((H, W), dtype=bool)
                mask_count = 0
                while mask_count < high:
                    max_mask_patches = high - mask_count

                    delta = 0
                    for attempt in range(10):
                        low = (min(H, W) // 3) ** 2 
                        target_area = random.uniform(low, max_mask_patches)
                        aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                        h = int(round(math.sqrt(target_area * aspect_ratio)))
                        w = int(round(math.sqrt(target_area / aspect_ratio)))
                        if w < W and h < H:
                            top = random.randint(0, H - h)
                            left = random.randint(0, W - w)

                            num_masked = mask[top: top + h, left: left + w].sum()
                            if 0 < h * w - num_masked <= max_mask_patches:
                                for i in range(top, top + h):
                                    for j in range(left, left + w):
                                        if mask[i, j] == 0:
                                            mask[i, j] = 1
                                            delta += 1

                        if delta > 0:
                            break

                    if delta == 0:
                        break
                    else:
                        mask_count += delta
            
            elif self.pred_shape == 'rand':
                mask = np.hstack([
                    np.zeros(H * W - int(high)),
                    np.ones(int(high)),
                ]).astype(bool)
                np.random.shuffle(mask)
                mask = mask.reshape(H, W)

            else:
                # no implementation
                assert False

            mask_key = 'msk_' + ikey
            data[mask_key] = mask
        return data




# On each spawned worker
def worker(args):
    rank = args.local_rank
    print(args)
    print('\n\n\n============os environ==========')
    print(os.environ)
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    print(rank, world_size)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    model = nn.Linear(1, 1, bias=False).to(rank)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[rank], output_device=rank
    )

    train_tfr_list =sorted(file_list)[:6]
    ds = TFRecords(density_loader, train_tfr_list, 100)



    start_t = timeit.default_timer()
    loader = torch.utils.data.DataLoader(ds, batch_size=24, pin_memory=True)
    print(len(loader))
    for i, data in enumerate(loader):
        print('rank: ', rank, 'batch_index: ', i, '   patch_index: ', data['patch_index'])
        time.sleep(0.5)
    end_t = timeit.default_timer()
    print(f"loading time: ", end_t - start_t)
    # Rank 1 gets one more input than rank 0.e 
    inputs = [torch.tensor([1]).float() for _ in range(10 + rank)]
    with model.join():
        for _ in range(5):
            for inp in inputs:
                loss = model(inp).sum()
                loss.backward()
    # Without the join() API, the below synchronization will hang
    # blocking for rank 1's allreduce to complete.
    torch.cuda.synchronize(device=rank)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-rank", type=int)
    args = parser.parse_args()
    
    # print("\n===============main==============\n")
    # print(args)
    # print("\n===============END OF main==============\n")
    worker(args)


#############
#% Excution 
#  python -m torch.distributed.launch --nproc_per_node=1 cellino_tfdata_loader.py