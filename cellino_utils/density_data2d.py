import numpy as np
import tensorflow as tf
# from zarrtfutil.tfutil import TFRreader
from  ml_util.zarr_tfr_writer import TFRreader

class DensityData2D:
    def __init__(self, filenames, nchannel_in_data, patch_shape,
                 multiscale_list, density_scale,
                 cutoff_overgrown, cutoff_empty,
                 downsample_method=None,
                 zindices=[0], proc_func=None, dataready=True,
                 nrepeat=1, shuffle_buffer_size=512, drop_remainder=True,
                 batch_size=5):
        """ DensityData2D data loader
            data (channel x height x width)
            to prepare NN inputs with channel last and
            to make the obj to be ready for iter()
            USE: 
                data = DensityData2D(tfr_list,...)
                iterdata = iter(data)

            NOTE: input data structure:
            self.features_shape = [(nchannel_in_data, height, width), (1, height, width), (1,)]
            self.features_name = ['X', 'Y', 'patch_index']
            self.feature_data_type = [tf.float32, tf.float32, tf.int64]

        Args:
            filenames (list): list of TFrecords
            nchannel_in_data (int): # channel of data
            patch_shape (tuple): 2D tuple for image patch shape
            multiscale_list (list): # multi-scale level list
            density_scale(float): scaling density
            cutoff_overgrown(float): cutoff for overgrown area
            cutoff_empty(float): cutoff for empty area

            zindices (list): list of z index to select for channel.
            proc_func (_type_, optional): _description_. Defaults to None.
            dataready (bool, optional): bool to make data reday. Defaults to True.
            nrepeat (int, optional): repeat number of the data. Defaults to 1.
            shuffle buffer (int, optional): shuffle buffer size. Defaults to 512.
            drop_remainder (bool, optional): _description_. Defaults to True.
            batch_size (int, optional): _description_. Defaults to 5.
        """
        self.filenames = filenames
        self.nchannel_in_data = nchannel_in_data
        self.patch_shape = patch_shape
        self.batch_size = batch_size
        self.shuffle_buffer_size = shuffle_buffer_size

        height, width = self.patch_shape
        nchannel_in_data = self.nchannel_in_data

        self.density_scale = density_scale
        self.cutoff_overgrown = cutoff_overgrown
        self.cutoff_empty = cutoff_empty

        #NOTE: define input-data structure here ####################
        self.features_shape = [(nchannel_in_data, height, width), (1, height, width), (1,)]
        self.features_name = ['brt', 'density', 'patch_index']
        self.feature_data_type = [tf.float32, tf.float32, tf.int64]

        ###############################

        # calculate subsample grid for each resolution scale
        self.multiscale_list = multiscale_list
        self.downsample_method = downsample_method
        ss_grids = []
        for d in multiscale_list:
            grid = tuple(np.array([2**d, 2**d]))
            # sample x Y x X x channel
            # ss_grid = (slice(None),) + tuple(slice(0, None, g) for g in grid) + (slice(None),) 
            ss_grid = tuple(slice(0, None, g) for g in grid) + (slice(None),) 
            ss_grids.append(ss_grid)
        self.ss_grids = ss_grids

        self.zindices = zindices
        if proc_func is None:
            self.proc_func = self.form_inputs_outputs #self._extract_zslice_and_channel_last
        else:
            self.proc_func = proc_func

        self.dataready = False

        if dataready:
            dataset = self.load_data()
            dset = dataset.map(self.proc_func)
            if self.shuffle_buffer_size > 0:
                dset = dset.shuffle(self.shuffle_buffer_size)
            dset = dset.batch(self.batch_size, drop_remainder=drop_remainder)

            self.tfdata = dset.repeat(nrepeat).prefetch(buffer_size=-1)
            self.dataready = True
        else:
            print("data not loaded yet. The 'else' condition is for debug purpose of different type of processing.")
            

    def _extract_zslice_and_channel_last(self, dataset):
        """ extract z slices selected and reform (zs x height x width) to (height x width x zs)
        """
        X_name = self.features_name[0]
        Y_name = self.features_name[1]
        X, Y = dataset[X_name], dataset[Y_name]
        X = [X[z] for z in self.zindices]
        X = tf.stack(X)
        X = tf.transpose(X, (1, 2, 0))
        Y = tf.transpose(Y, (1, 2, 0))
        return X, Y
    
    def form_inputs_outputs(self, dataset):
        X, D = self._extract_zslice_and_channel_last(dataset)

        scaling_method = self.downsample_method
        outputs = {}
        for ss_grid, scale_name in zip(self.ss_grids, self.multiscale_list):
            if ss_grid[0].step is None:
                scale = self.density_scale
            else:
                scale = self.density_scale * ss_grid[0].step * ss_grid[1].step
            # mD.append(D[ss_grid]*scale)
            if scaling_method == "mean":
                ss_grid1 = (ss_grid[0].step, ss_grid[1].step)
                outputs[f'den_{scale_name}'] = downscale_mean(D, ss_grid1) * scale
            else: # sampling on the grid
                outputs[f'den_{scale_name}'] = D[ss_grid] * scale 

        cutoff_overgrown = tf.constant(self.cutoff_overgrown, dtype='float32')
        cutoff_empty = tf.constant(self.cutoff_empty, dtype='float32')

        D_ = tf.squeeze(D)
        # find empty area, overgrown, normal area
        inds_empty = tf.where(D_ <= cutoff_empty)
        inds_over = tf.where(D_ >= cutoff_overgrown)
        inds_normal = tf.where(tf.logical_and(D_ < cutoff_overgrown , D_ > cutoff_empty))          

        # create pixel indices in D_.shape + (3, )  
        slice_normal, slice_overgrown = 1, 2 #slice to fill, empty: [..., 0], normal: [..., 1], over: [..., 2] 

        # create class masks
        area_cat = tf.zeros(D_.shape + (3, ), dtype='int64')
        n_empty = tf.shape(inds_empty)[0] 
        n_normal = tf.shape(inds_normal)[0]
        n_over = tf.shape(inds_over)[0]#.shape[0]

        inds_empty_ = tf.concat([inds_empty, tf.zeros((n_empty, 1), dtype='int64')], axis=-1)
        area_cat = tf.tensor_scatter_nd_update(area_cat, inds_empty_, tf.ones((n_empty,), dtype='int64'))

        inds_normal_ = tf.concat([inds_normal, slice_normal * tf.ones((n_normal, 1), dtype='int64')], axis=-1)
        area_cat = tf.tensor_scatter_nd_update(area_cat, inds_normal_, tf.ones((n_normal,), dtype='int64'))

        inds_over_ = tf.concat([inds_over, slice_overgrown * tf.ones((n_over, 1), dtype='int64')], axis=-1)
        area_cat = tf.tensor_scatter_nd_update(area_cat, inds_over_, tf.ones((n_over,), dtype='int64'))

        outputs['prob_class'] = area_cat

        return X, outputs

    def load_data(self):
        reader = TFRreader(self.filenames)
        reader.set_feat(self.features_shape, self.features_name, self.feature_data_type)
        func = reader.gen_parsefun(self.features_name, self.features_shape)
        dataset = tf.data.TFRecordDataset(self.filenames).map(func)
        return dataset




def downscale_mean(image, ss_grid):
    """ downscale image with averaging pixels within kernel,
        image: 3D, HWC
        ss_grid: tuple, vertical x horizontal grid
    """
    
    image = tf.expand_dims(image, axis=0)
    filter_shape = ss_grid
    area = tf.constant(filter_shape[0] * filter_shape[1], dtype=image.dtype)

    filter_shape += (tf.shape(image)[-1],1)
    kernel = tf.ones(shape=filter_shape, dtype=image.dtype)

    strides = (1,) + ss_grid + (1,)
    output = tf.nn.depthwise_conv2d(
        image, kernel, strides=strides, padding="VALID"
    )

    output /= area
    return output[0]