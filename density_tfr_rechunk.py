#%% tfrecords rechunk test
import os
import glob
from pathlib import Path
import random
from cellino_tfdata_loader import WB_DIR, rechunk_tfrecords, density_original_loader



artifact_name = 'TFRECORD-4x_density'
data_path = os.path.join(WB_DIR, artifact_name)
file_list = [glob.glob(os.path.join(data_path, '*-HIGH.tfr')),
             glob.glob(os.path.join(data_path, '*-MED.tfr')),
             glob.glob(os.path.join(data_path, '*-LOW.tfr')),
             glob.glob(os.path.join(data_path, '*-EDGE.tfr'))]



# new_data_path = Path('/media/slee/DATA1/DL/data/TFRECORD-4x_density/') / 'rechunk'
new_data_path = Path('/home/slee_cellinobio_com/dl-data1/data/TFRECORD-4x_density/') / 'rechunk'
ntest =  4

tr_tfr_list, test_tfr_list = [], []
for list1 in file_list:
    tr_tfr_list.extend(list1[:-ntest])
    test_tfr_list.extend(list1[-ntest:])


list1 = [x.split('/')[-1] for x in tr_tfr_list]
list1 = sorted(list1)
tr_tfr_list = [os.path.join(data_path, x) for x in list1]

if not os.path.exists(str(new_data_path)): 
    os.makedirs(str(new_data_path))
    
train_data_fntempl = str(new_data_path / 'train')
test_data_fntempl = str(new_data_path / 'test')

tfr_loader1 = density_original_loader(tr_tfr_list)
nchunk1, last_chunk_sample1 = rechunk_tfrecords(tfr_loader1, train_data_fntempl, 500, write_tfrinfo=True)

tfr_loader2 = density_original_loader(test_tfr_list)
nchunk2, last_chunk_sample2 = rechunk_tfrecords(tfr_loader2, test_data_fntempl, 500, write_tfrinfo=True)


# tfr_loader = density_loader([tr_tfr_list[0]])

# for i, x in enumerate(tfr_loader.load_data().as_numpy_iterator()):
#     print(i)



#%% 
##################################################
## validating new tfrecords 

# import tensorflow as tf
# from cellino_tfdata_loader import new_density_loader


# x = new_density_loader(["test_0.tfr"])
# x = x.load_data().as_numpy_iterator()

# def drop_tfr_name(data):
#     out_data = dict()
#     for key, val in data.items():
#         if key != 'tfr_name':
#             out_data[key] = val
#     return out_data

# from cellino_tfdata_loader import TFRecords
# import torch
# ds = TFRecords(new_density_loader, ["test_0.tfr"], nsample_per_file=100, process_fcn_list=[drop_tfr_name], shuffle_buffer_size=0)
# data_loader = torch.utils.data.DataLoader(ds, batch_size=24, pin_memory=True)


# for i, batch in enumerate(data_loader):
#     if i == 0:
#         break
#     print(i)


