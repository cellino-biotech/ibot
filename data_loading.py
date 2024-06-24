import os
import wandb
import pandas as pd
import glob
import tensorflow
import torch



from cellino_utils.density_data2d import DensityData2D

# WB_DIR = 'mnt/disks/dl-data1/wandb'
WB_DIR = '/Users/sangkyunlee/Cellino/DL/data/wandb'
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


file_list = glob.glob(os.path.join(art_dir, '*.tfr'))




train_tfr_list =sorted(file_list)[:5]
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



train_data_loader = DensityData2D(train_tfr_list,
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

# tfdata = train_data_loader.tfdata
# for i, dat in enumerate(tfdata.as_numpy_iterator()):
#     print(i, f'data_shape: {dat[0].shape}')



rawdata = train_data_loader.load_data()
# for i, dat in enumerate(rawdata.as_numpy_iterator()):
#     print(i)

# from torch.utils.data import DataLoader
# a = next(rawdata.as_numpy_iterator())


class tfdata_wrapper:
    def __init__(self, tf_data, feat_codes={'X': 'brt', 'Y': 'density'}):
        self.tf_data = tf_data.as_numpy_iterator()
        self.feat_codes = feat_codes
    
    def __getitem__(self, idx):
        dat = next(self.tf_data)
        X = dat[self.feat_codes['X']]
        Y = dat[self.feat_codes['Y']]
        return X, Y


density_data = tfdata_wrapper(rawdata)


d1 = next(iter(density_data))
