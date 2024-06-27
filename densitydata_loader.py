import os
import math
import wandb
import pandas as pd
import glob
import tensorflow
import torch
from functools import *
import timeit
import time

from cellino_utils.density_data2d import DensityData2D

WB_DIR = 'mnt/disks/dl-data1/wandb'
# WB_DIR = '/Users/sangkyunlee/Cellino/DL/data/wandb'
# WB_DIR = '/media/slee/DATA1/DL/data/wandb'
wb_entity = 'cellino-ml-ninjas'
wb_project_name = 'WP_CELL_ID-4x_density'
artifact_name = 'TFRECORD-4x_density'
ver = 'latest'

api = wandb.Api()
artifact = api.artifact(f'{wb_entity}/{wb_project_name}/{artifact_name}:{ver}')#, type='dataset_with_coarse_clean')
art_dir = artifact.download(root=os.path.join(WB_DIR, artifact_name))

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





density_loader  = partial(DensityData2D,
                          nchannel_in_data=nchannel_in_data,
                        patch_shape=patch_shape,
                        dataready=dataready,
                        multiscale_list=multiscale_list,
                        density_scale=density_scale,
                        cutoff_overgrown=cutoff_overgrown,
                        cutoff_empty=cutoff_empty,
                        nrepeat=nrepeat,
                        zindices=z_indices,
                        shuffle_buffer_size=shuffle_buffer_size,
                        drop_remainder=drop_remainder,
                        batch_size=train_batch_size)



#%%
import argparse
import torch
import torch.distributed as dist
import os
import torch.nn as nn




class TFDataset(torch.utils.data.IterableDataset):
    def __init__(self, loader, list_tfrs,  shuffle_buffer_size=512, feat_codes={'X': 'brt', 'Y': 'density'}):
        super(TFDataset).__init__()

        overall_start = 0
        overall_end = len(list_tfrs)
        worker_id = int(os.environ['RANK'])
        num_workers = int(os.environ['WORLD_SIZE'])
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
        # self.prefetch = 
        
        # self.sample_count = 0
    def __iter__(self):

        subloader = self.loader(self.list_files[self.start:self.end])
        return subloader.load_data().shuffle(self.shuffle_buffer_size)).as_numpy_iterator()
    

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
    ds = TFDataset(density_loader, train_tfr_list)



    start_t = timeit.default_timer()
    loader = torch.utils.data.DataLoader(ds, batch_size=24, pin_memory=True)
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
#  python -m torch.distributed.launch --nproc_per_node=2 pytorch_loader.py