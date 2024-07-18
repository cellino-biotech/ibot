#%% tfrecords rechunk test
import os
import glob
from pathlib import Path

from cellino_tfdata_loader import WB_DIR, rechunk_tfrecords, density_loader



artifact_name = 'TFRECORD-4x_density'
data_path = os.path.join(WB_DIR, artifact_name)
file_list = [glob.glob(os.path.join(data_path, '*-HIGH.tfr')),
             glob.glob(os.path.join(data_path, '*-MED.tfr')),
             glob.glob(os.path.join(data_path, '*-LOW.tfr')),
             glob.glob(os.path.join(data_path, '*-EDGE.tfr'))]


ntest =  4

tr_tfr_list, test_tfr_list = [], []
for list1 in file_list:
    tr_tfr_list.extend(list1[:-ntest])
    test_tfr_list.extend(list1[-ntest:])


new_data_path = Path('/media/slee/DATA1/DL/data/TFRECORD-4x_density/') / 'rechunk'
if not os.path.exists(str(new_data_path)): 
    os.makedirs(str(new_data_path))
    
train_data_fntempl = str(new_data_path / 'train')
test_data_fntempl = str(new_data_path / 'test')

tfr_loader1 = density_loader(tr_tfr_list)
nchunk1, last_chunk_sample1 = rechunk_tfrecords(tfr_loader1, train_data_fntempl, 500)

tfr_loader2 = density_loader(test_tfr_list)
nchunk2, last_chunk_sample2 = rechunk_tfrecords(tfr_loader2, test_data_fntempl, 500)


# tfr_loader = density_loader(['test_3.tfr'])

# for i, x in enumerate(tfr_loader.load_data().as_numpy_iterator()):
#     print(i)
