import numpy as np
np.random.seed(27)
import tensorflow as tf
tf.random.set_seed(27)
from tensorflow.keras import layers
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
from tqdm.keras import TqdmCallback
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import clear_output
import time


print("Tensorflow Version: ", tf.__version__)
print("Numpy Version: ", np.__version__)

# Define Self-Supervised Auxiliary Tasks:
def prepare_domain_detection_dataset(x1, x2):
    """
    Takes two different domain unlabeled datasets and produce a domain detection task dataset.
    
    Arguments:
    x1 -- samples from the first domain
    x2 -- samples from the second domain
    
    Returns:
    X -- a joint dataset with both x1 and x2
    Y -- a label set with label of zero for x1 samples and label of one for x2 samples
    """
    y1 = np.zeros((x1.shape[0],), dtype='float32')
    y2 = np.ones((x2.shape[0],), dtype='float32')
    X = np.concatenate([x1, x2])
    Y = np.concatenate([y1, y2])
    return X, Y
    
def prepare_rotation_detection_dataset(x):
    """
    Takes unlabeled dataset x and produce a rotation detection task. samples get rotated on time axis.
    
    Arguments:
    x -- unlabeled samples with shape (Batch, Timestep, Features)
    
    Returns:
    X -- a joint dataset of normal and rotated samples on time axis
    Y -- a label set with label of zero for normal samples and label of one for rotated samples
    """
    
    x_rot = x[:,::-1,:]
    y_rot = np.ones((x_rot.shape[0],), dtype='float32')
    x_not_rot = x
    y_not_rot = np.zeros((x_not_rot.shape[0],), dtype='float32')
    
    X = np.concatenate([x_rot, x_not_rot])
    Y = np.concatenate([y_rot, y_not_rot])
    
    return X, Y


def prepare_time_segment_detection_dataset(x, cutoff=2):
    """
    Takes unlabeled dataset x and produce a time-segment detection task. samples get shattered on time axis in equally sized segments based on cutoff value and get repeated time-steps/cutoff times on the time axis.
    
    Arguments:
    x -- unlabeled samples with shape (Batch, Timestep, Features)
    
    Returns:
    X -- a joint dataset of each time-segment based on cutoff value
    Y -- a labels set with labels of 0 to (int(time-steps/cutoff) - 1) for each sample base on the time-segment it represents
    """
    
    repeat = x.shape[1] // cutoff
    label = 0.
    for i in range(repeat):
        x_seg = np.repeat(x[:, cutoff*i:cutoff*(i+1) ,:], repeat, axis=1)
        y_seg = np.zeros((x_seg.shape[0],), dtype='float32') + label
        if label == 0:
            X = x_seg
            Y = y_seg
        else:
            X = np.concatenate([X, x_seg], axis=0)
            Y = np.concatenate([Y, y_seg], axis=0)
        label += 1

    return X, Y


def prepare_band_detection_dataset(x):
    """
    Takes unlabeled dataset x and produce a spectral band detection task. samples get shattered on feature axis into single feature vectors and get repeated #Features times on the feature axis.
    
    Arguments:
    x -- unlabeled samples with shape (Batch, Timestep, Features)
    
    Returns:
    X -- a joint dataset of each spectral band
    Y -- a labels set with labels of 0 to #Features for each sample base on the spectral band it represents
    """
    
    repeat = 7
    label = 0.
    for i in range(repeat):
        x_band = np.repeat(x[:, :, i:(i+1)], repeat, axis=2)
        y_band = np.zeros((x_band.shape[0],), dtype='float32') + label
        if label == 0:
            X = x_band
            Y = y_band
        else:
            X = np.concatenate([X, x_band], axis=0)
            Y = np.concatenate([Y, y_band], axis=0)
        label += 1

    return X, Y


def prepare_all_auxiliary_dataset(x_source, y_source, x_unlabel_target):
    """
    Takes unlabeled dataset x_unlabel and labeled dataset (x, y) and produce a joint dataset for all of the auxiliary tasks based on x_unlabel_target and x and also the main supervised task base on (x, y) which are from the source domain. This is a single dataset with all tasks.
    
    Arguments:
    x_source -- samples from source domain
    y_source -- labels of samples from source domain
    x_unlabel_target -- unlabeled samples with shape (Batch, Timestep, Features) from target domain
    
    Returns:
    X -- a joint dataset of all auxiliary tasks
    y_cls -- labels of crop classification task with an extra label to ignore samples in X which are not for crop task
    y_bands -- labels of band classification task with an extra label to ignore samples in X which are not for band task
    y_seg -- labels of time-segment classification task with an extra label to ignore samples in X which are not for time-segment task
    y_rot -- labels of rotation classification task with an extra label to ignore samples in X which are not for rotation task
    y_dom -- labels of domain classification task with an extra label to ignore samples in X which are not for domain task
    """
    
    # crop classification task
    X = x_source.copy()
    y_cls = y_source.copy()
    n1 = y_cls.shape[0]

    # domain detection dataset
    x_aux, y_dom = prepare_domain_detection_dataset(x_source, x_unlabel_target)
    n2 = y_dom.shape[0]
    X = np.concatenate([X, x_aux], axis=0)
    
    # ********** Free some memory here, I had some problems **********
    x_unlabel = np.concatenate([x_source, x_unlabel_target], axis=0) 
    del x_source, y_source, x_unlabel_target
    # ****************************************************************
    
    # bands detection dataset from unlabeled samples
    x_aux, y_band = prepare_band_detection_dataset(x_unlabel)
    n3 = y_band.shape[0]
    X = np.concatenate([X, x_aux], axis=0) # I did it this way for lack of enough memory on my system
    
    # time-segment detection dataset from unlabeled samples
    x_aux, y_seg = prepare_time_segment_detection_dataset(x_unlabel)
    n4 = y_seg.shape[0]
    X = np.concatenate([X, x_aux], axis=0)
    
    # rotation detection dataset from unlabeled samples
    x_aux, y_rot = prepare_rotation_detection_dataset(x_unlabel)
    n5 = y_rot.shape[0]
    X = np.concatenate([X, x_aux], axis=0)
    del x_aux
    
    # order to concat ==> Crop -> Dom -> Band -> Segment -> Rotation

    # ignore value for crop classification is 4:
    y_dummy = np.zeros((n2 + n3 + n4 + n5,), dtype='float32') + 4.0
    y_cls = np.concatenate([y_cls, y_dummy])
    
    # ignore value for domain classification is 2:
    y_dummy = np.zeros((n1,), dtype='float32') + 2.0
    y_dom = np.concatenate([y_dummy, y_dom])
    y_dummy = np.zeros((n3 + n4 + n5,), dtype='float32') + 2.0
    y_dom = np.concatenate([y_dom, y_dummy])

    # ignore value for band classification is 7:
    y_dummy = np.zeros((n1 + n2,), dtype='float32') + 7.0
    y_band = np.concatenate([y_dummy, y_band])
    y_dummy = np.zeros((n4 + n5,), dtype='float32') + 7.0
    y_band = np.concatenate([y_band, y_dummy])

    # ignore value for time-segment classification is 5:
    y_dummy = np.zeros((n1 + n2 + n3,), dtype='float32') + 5.0
    y_seg = np.concatenate([y_dummy, y_seg])
    y_dummy = np.zeros((n5,), dtype='float32') + 5.0
    y_seg = np.concatenate([y_seg, y_dummy])

    # ignore value for rotation classification is 2:
    y_dummy = np.zeros((n1 + n2 + n3 + n4,), dtype='float32') + 2.0
    y_rot = np.concatenate([y_dummy, y_rot])
     
    # sanity checks:
    # crop classification labels:
    np.testing.assert_equal(y_cls[n1:], 4)
    # domain classification labels:
    np.testing.assert_equal(y_dom[:n1], 2)
    np.testing.assert_equal(y_dom[n1+n2:], 2)
    # band classification labels:
    np.testing.assert_equal(y_band[:n1+n2], 7)
    np.testing.assert_equal(y_band[n1+n2+n3:], 7)
    # time-segment classification labels:
    np.testing.assert_equal(y_seg[:n1+n2+n3], 5)
    np.testing.assert_equal(y_seg[n1+n2+n3+n4:], 5)
    # rotation classification labels:
    np.testing.assert_equal(y_rot[:n1+n2+n3+n4], 2)
    
    return X, y_cls, y_band, y_seg, y_rot, y_dom


def prepare_all_auxiliary_dataset_no_dom(x_source, y_source, x_unlabel_target):
    """
    Takes unlabeled dataset x_unlabel and labeled dataset (x, y) and produce a joint dataset for all of the auxiliary tasks based on x_unlabel_target and x and also the main supervised task base on (x, y) which are from the source domain. This is a single dataset with all tasks.
    
    Arguments:
    x_source -- samples from source domain
    y_source -- labels of samples from source domain
    x_unlabel_target -- unlabeled samples with shape (Batch, Timestep, Features) from target domain
    
    Returns:
    X -- a joint dataset of all auxiliary tasks
    y_cls -- labels of crop classification task with an extra label to ignore samples in X which are not for crop task
    y_bands -- labels of band classification task with an extra label to ignore samples in X which are not for band task
    y_seg -- labels of time-segment classification task with an extra label to ignore samples in X which are not for time-segment task
    y_rot -- labels of rotation classification task with an extra label to ignore samples in X which are not for rotation task
    y_dom -- labels of domain classification task with an extra label to ignore samples in X which are not for domain task
    """
    
    # crop classification task
    X = x_source.copy()
    y_cls = y_source.copy()
    n1 = y_cls.shape[0]

    # ********** Free some memory here, I had some problems **********
    x_unlabel = np.concatenate([x_source, x_unlabel_target], axis=0) 
    del x_source, y_source, x_unlabel_target
    # ****************************************************************
    
    # bands detection dataset from unlabeled samples
    x_aux, y_band = prepare_band_detection_dataset(x_unlabel)
    n2 = y_band.shape[0]
    X = np.concatenate([X, x_aux], axis=0) # I did it this way for lack of enough memory on my system
    
    # time-segment detection dataset from unlabeled samples
    x_aux, y_seg = prepare_time_segment_detection_dataset(x_unlabel)
    n3 = y_seg.shape[0]
    X = np.concatenate([X, x_aux], axis=0)
    
    # rotation detection dataset from unlabeled samples
    x_aux, y_rot = prepare_rotation_detection_dataset(x_unlabel)
    n4 = y_rot.shape[0]
    X = np.concatenate([X, x_aux], axis=0)
    del x_aux
    
    # order to concat ==> Crop -> Band -> Segment -> Rotation

    # ignore value for crop classification is 4:
    y_dummy = np.zeros((n2 + n3 + n4,), dtype='float32') + 4.0
    y_cls = np.concatenate([y_cls, y_dummy])

    # ignore value for band classification is 7:
    y_dummy = np.zeros((n1,), dtype='float32') + 7.0
    y_band = np.concatenate([y_dummy, y_band])
    y_dummy = np.zeros((n3 + n4,), dtype='float32') + 7.0
    y_band = np.concatenate([y_band, y_dummy])

    # ignore value for time-segment classification is 5:
    y_dummy = np.zeros((n1 + n2,), dtype='float32') + 5.0
    y_seg = np.concatenate([y_dummy, y_seg])
    y_dummy = np.zeros((n4,), dtype='float32') + 5.0
    y_seg = np.concatenate([y_seg, y_dummy])

    # ignore value for rotation classification is 2:
    y_dummy = np.zeros((n1 + n2 + n3,), dtype='float32') + 2.0
    y_rot = np.concatenate([y_dummy, y_rot])
     
    # sanity checks:
    # crop classification labels:
    np.testing.assert_equal(y_cls[n1:], 4)
    # band classification labels:
    np.testing.assert_equal(y_band[:n1], 7)
    np.testing.assert_equal(y_band[n1+n2:], 7)
    # time-segment classification labels:
    np.testing.assert_equal(y_seg[:n1+n2], 5)
    np.testing.assert_equal(y_seg[n1+n2+n3:], 5)
    # rotation classification labels:
    np.testing.assert_equal(y_rot[:n1+n2+n3], 2)
    
    return X, y_cls, y_band, y_seg, y_rot

def prepare_only_domain_baseline_dataset(x_source, y_source, x_unlabel_target):
    """
    Takes unlabeled dataset x_unlabel and labeled dataset (x, y) and produce a joint dataset for all of the auxiliary tasks based on x_unlabel_target and x and also the main supervised task base on (x, y) which are from the source domain. This is a single dataset with all tasks.
    
    Arguments:
    x_source -- samples from source domain
    y_source -- labels of samples from source domain
    x_unlabel_target -- unlabeled samples with shape (Batch, Timestep, Features) from target domain
    
    Returns:
    X -- a joint dataset of all auxiliary tasks
    y_cls -- labels of crop classification task with an extra label to ignore samples in X which are not for crop task
    y_dom -- labels of domain classification task with an extra label to ignore samples which are not for domain task
    """
    
    # crop classification task
    X = x_source.copy()
    y_cls = y_source.copy()
    n1 = y_cls.shape[0]

    # domain detection dataset
    x_aux, y_dom = prepare_domain_detection_dataset(x_source, x_unlabel_target)
    n2 = y_dom.shape[0]
    X = np.concatenate([X, x_aux], axis=0)

    # ignore value for crop classification is 4:
    y_dummy = np.zeros((n2,), dtype='float32') + 4.0
    y_cls = np.concatenate([y_cls, y_dummy])
    
    # ignore value for domain classification is 2:
    y_dummy = np.zeros((n1,), dtype='float32') + 2.0
    y_dom = np.concatenate([y_dummy, y_dom])
    
    # sanity checks:
    # crop classification labels:
    np.testing.assert_equal(y_cls[n1:], 4)
    # domain classification labels:
    np.testing.assert_equal(y_dom[:n1], 2)
    
    return X, y_cls, y_dom


# Define custome loss functions for auxiliary tasks to be able to train everything jointly
class MaskedBinaryAccuracyMetric(tf.keras.metrics.Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs) # handles base args (e.g., dtype)
        self.total = self.add_weight("total", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")   
   
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.squeeze(y_true)
        mask = tf.where(y_true!=2)
        y_true = tf.gather_nd(y_true, mask)
        y_pred = tf.gather_nd(y_pred, mask)
        y_true = tf.squeeze(y_true)
        y_pred = tf.squeeze(y_pred)
        y_pred = tf.where(tf.less(y_pred, 0.5), 0., 1.)
        metric = tf.cast(tf.equal(y_true, y_pred), dtype=tf.float32)
        self.total.assign_add(tf.reduce_sum(metric))
        self.count.assign_add(tf.cast(tf.size(metric), tf.float32)) 

    def result(self):
        return self.total / self.count
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config}

class MaskedBXE(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def call(self, y_true, y_pred):
        y_true = tf.squeeze(y_true)
        mask = tf.where(y_true!=2)
        y_true = tf.gather_nd(y_true, mask)
        y_pred = tf.gather_nd(y_pred, mask)
        y_true = tf.squeeze(y_true)
        y_pred = tf.squeeze(y_pred)
        loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        return tf.reduce_mean(loss)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config}

class MaskedCXE(tf.keras.losses.Loss):
    def __init__(self, ignore=None, **kwargs):
        self.ignore = ignore
        super().__init__(**kwargs)
    def call(self, y_true, y_pred):
        y_true = tf.squeeze(y_true)
        mask = tf.where(y_true!=self.ignore)
        y_true = tf.gather_nd(y_true, mask)
        y_pred = tf.gather_nd(y_pred, mask)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        return tf.reduce_mean(loss)
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "ignore": self.ignore}

class MaskedCategoricalAccuracyMetric(tf.keras.metrics.Metric):
    def __init__(self, ignore=None, **kwargs):
        super().__init__(**kwargs) # handles base args (e.g., dtype)
        self.total = self.add_weight("total", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")
        self.ignore = ignore     
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.squeeze(y_true)
        mask = tf.where(y_true!=self.ignore)
        y_true = tf.gather_nd(y_true, mask)
        y_pred = tf.gather_nd(y_pred, mask)
        y_pred = tf.cast(tf.argmax(y_pred, axis=1), dtype=tf.float32)
        y_pred = tf.squeeze(y_pred)
        metric = tf.cast(tf.equal(y_true, y_pred), dtype=tf.float32)
        self.total.assign_add(tf.reduce_sum(metric))
        self.count.assign_add(tf.cast(tf.size(metric), tf.float32))   
    def result(self):
        return self.total / self.count
    def get_config(self):
        base_config = super().get_config()
        return {**base_config}


# Define gradient reversal layer
@tf.custom_gradient
def gradient_reversal(x):
    out = tf.identity(x)
    def grad_fn(dy):
        return -1. * dy
    return out, grad_fn

class GradientReversalLayer(layers.Layer):
    def __init__(self):
        super(GradientReversalLayer, self).__init__()

    def call(self, x):
        return gradient_reversal(x)

# Define helper functions for testing
def test_base_model(x, y, model=None, save_name=None):
    test_ds = tf.data.Dataset.from_tensor_slices(x).batch(2**14).prefetch(tf.data.experimental.AUTOTUNE)
    y_pred = []
    for x_s in test_ds:
        yhat = model(x_s, training=False)
        y_pred.append(np.argmax(yhat, axis=-1))
            
    y_pred = np.concatenate(y_pred)
    acc = np.sum(y == y_pred) / y.shape[0]
    
    target_names = ['wheat', 'corn', 'rice', 'other']
    print(classification_report(y, y_pred, target_names=target_names))
    cm = confusion_matrix(y, y_pred, normalize='true')
    cm_display = ConfusionMatrixDisplay(cm, display_labels=target_names).plot(cmap=plt.cm.Blues)
    plt.grid(False)
   
    if save_name is not None:
        plt.savefig(save_name)
    plt.show()
    return acc

def test_aux_tasks_model(x, y, model=None, name=None):
    test_ds = tf.data.Dataset.from_tensor_slices(x).batch(2**14).prefetch(tf.data.experimental.AUTOTUNE)
    y_pred = []
    for x_s in test_ds:
        yhat,_,_,_,_ = model(x_s, training=False)
        y_pred.append(np.argmax(yhat, axis=-1))
            
    y_pred = np.concatenate(y_pred)
    acc = np.sum(y == y_pred) / y.shape[0]
    
    target_names = ['wheat', 'corn', 'rice', 'other']
    print(classification_report(y, y_pred, target_names=target_names))
    cm = confusion_matrix(y, y_pred, normalize='true')
    cm_display = ConfusionMatrixDisplay(cm, display_labels=target_names).plot(cmap=plt.cm.Blues)
    plt.grid(False)
    if name is not None:
        plt.savefig(name)
    plt.show()
    return acc

def test_aux_tasks_model_no_dom(x, y, model=None, name=None):
    test_ds = tf.data.Dataset.from_tensor_slices(x).batch(2**14).prefetch(tf.data.experimental.AUTOTUNE)
    y_pred = []
    for x_s in test_ds:
        yhat,_,_,_ = model(x_s, training=False)
        y_pred.append(np.argmax(yhat, axis=-1))
            
    y_pred = np.concatenate(y_pred)
    acc = np.sum(y == y_pred) / y.shape[0]
    
    target_names = ['wheat', 'corn', 'rice', 'other']
    print(classification_report(y, y_pred, target_names=target_names))
    cm = confusion_matrix(y, y_pred, normalize='true')
    cm_display = ConfusionMatrixDisplay(cm, display_labels=target_names).plot(cmap=plt.cm.Blues)
    plt.grid(False)
    if name is not None:
        plt.savefig(name)
    plt.show()
    return acc

def test_aux_tasks_model_only_domain(x, y, model=None, name=None):
    test_ds = tf.data.Dataset.from_tensor_slices(x).batch(2**14).prefetch(tf.data.experimental.AUTOTUNE)
    y_pred = []
    for x_s in test_ds:
        yhat,_ = model(x_s, training=False)
        y_pred.append(np.argmax(yhat, axis=-1))
            
    y_pred = np.concatenate(y_pred)
    acc = np.sum(y == y_pred) / y.shape[0]
    
    target_names = ['wheat', 'corn', 'rice', 'other']
    print(classification_report(y, y_pred, target_names=target_names))
    cm = confusion_matrix(y, y_pred, normalize='true')
    cm_display = ConfusionMatrixDisplay(cm, display_labels=target_names).plot(cmap=plt.cm.Blues)
    plt.grid(False)
    if name is not None:
        plt.savefig(name)
    plt.show()
    return acc