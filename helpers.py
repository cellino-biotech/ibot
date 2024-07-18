
import numpy as np
import tensorflow as tf



class cellinoTFRreader:
    """ TFRecord reader with varLengFeature"""
    def __init__(self, 
                 tfr_name_list: list,
                 features_shape: list,
                 features_name: list,
                 features_data_type: list):
        self.tfr_name_list = tfr_name_list
        self.features_shape = features_shape
        self.features_name = features_name
        self.features_data_type = features_data_type
        self.feat_desc = {}

    def set_feat(self):
        N = len(self.features_shape)
        assert (len(self.features_name) == N and\
                len(self.features_data_type) == N), "feature specification not matched"
        
        feats_size = []
        for ps in self.features_shape:
            if ps is None:
                feats_size.append(None)
            else:
                feats_size.append(np.array(ps).prod())
        feat = {}
        for name, size, dtype in zip(self.features_name, feats_size, self.features_data_type):
            if size is None:
                feat[name] = tf.io.VarLenFeature(dtype=dtype)
            else:
                feat[name] = tf.io.FixedLenFeature([size], dtype=dtype)

        self.feat_desc = feat

    def _parse(self, _byte_data):
        parsed = tf.io.parse_single_example(_byte_data, self.feat_desc)
        out = {}
        for name, shape in zip(self.features_name, self.features_shape):
            if shape is None:
                out[name] = tf.sparse.to_dense(parsed[name])
            else:
                out[name] = tf.reshape(parsed[name], shape)
        return out

    # def gen_parsefun(self, self.features_name: list, self.features_shape: list):
    #     return partial(self._parse, feats_name, self.features_shape)
    
    def load_data(self):
        self.set_feat()
        dataset = tf.data.TFRecordDataset(self.tfr_name_list).map(self._parse)
        return dataset

