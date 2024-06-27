import os
import math
import wandb
import pandas as pd
import glob
import tensorflow
import torch
from functools import *


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



# train_data_loader = DensityData2D(train_tfr_list,
#                                nchannel_in_data=nchannel_in_data,
#                                patch_shape=patch_shape,
#                                dataready=dataready,
#                                multiscale_list=multiscale_list,
#                                density_scale=density_scale,
#                                cutoff_overgrown=cutoff_overgrown,
#                                cutoff_empty=cutoff_empty,
#                                nrepeat=nrepeat,
#                                zindices=z_indices,
#                                shuffle_buffer_size=shuffle_buffer_size,
#                                drop_remainder=drop_remainder,
#                                batch_size=train_batch_size)



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


class TFDataset(torch.utils.data.IterableDataset):
    def __init__(self, loader, list_tfrs, start, end, feat_codes={'X': 'brt', 'Y': 'density'}):
        super(TFDataset).__init__()
        assert end > start, "this example code only works with end >= start"
        self.start = start
        self.end = end
        assert end <= len(list_tfrs)
        self.feat_codes = feat_codes
        self.list_files = list_tfrs
        self.loader = loader
        
        # self.sample_count = 0
    def __iter__(self):

        subloader = self.loader(self.list_files[self.start:self.end])
        return subloader.load_data().as_numpy_iterator()


# train_tfr_list =sorted(file_list)[:2]
# ds = TFDataset(density_loader, train_tfr_list, start=0, end=len(train_tfr_list))

# for i, dat in enumerate(iter(ds)):
#     print(i)


# xx = list(torch.utils.data.DataLoader(ds, num_workers=0, batch_size=2))



# Define a `worker_init_fn` that configures each dataset copy differently
def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    overall_start = dataset.start
    overall_end = dataset.end
    # configure the dataset to only process the split workload
    per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
    worker_id = worker_info.id
    dataset.start = overall_start + worker_id * per_worker
    dataset.end = min(dataset.start + per_worker, overall_end)
    print(worker_info, dataset.start, dataset.end)

# Mult-process loading with the custom `worker_init_fn`
# Worker 0 fetched [3, 4].  Worker 1 fetched [5, 6].
xx=list(torch.utils.data.DataLoader(ds, num_workers=2, batch_size=2, worker_init_fn=worker_init_fn, pin_memory=True))
