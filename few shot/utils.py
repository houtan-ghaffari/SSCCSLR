import numpy as np
np.random.seed(27)
import tensorflow as tf
tf.random.set_seed(27)
from tensorflow.keras import layers
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

print("Numpy Version: ", np.__version__)
print("Tensorflow Version: ", tf.__version__)


def get_base_model():
    tf.keras.backend.clear_session()
    base_model = tf.keras.Sequential([
        layers.Input(shape=(10, 7)),
        layers.Conv1D(32, 1, name='conv_1'),
        layers.BatchNormalization(name='bn_1'),
        layers.LeakyReLU(name='act_1'),
        layers.LSTM(64, dropout=0.25, recurrent_dropout=0.25, return_sequences=True, name='lstm_1'),
        layers.LSTM(64, return_sequences=False, name='lstm_2'),
        layers.BatchNormalization(name='bn_2'),
        layers.Dense(32, name='dense_1'),
        layers.BatchNormalization(name='bn_3'),
        layers.LeakyReLU(name='act_2'),
        layers.Dropout(0.25, name='drop_1'),
        layers.Dense(4, activation='softmax', name='out')
    ])
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    base_model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    return base_model

def get_rot_model():
    tf.keras.backend.clear_session()
    input_x = layers.Input(shape=[10, 7])
    # trunk
    shared_x = layers.Conv1D(32, 1, name='conv_1')(input_x)
    shared_x = layers.BatchNormalization(name='bn_1')(shared_x)
    shared_x = layers.LeakyReLU(name='act_1')(shared_x)
    shared_x = layers.LSTM(64, dropout=0.25, recurrent_dropout=0.25, return_sequences=True, name='lstm_1')(shared_x)
    shared_x = layers.LSTM(64, return_sequences=False, name='lstm_2')(shared_x)
    shared_x = layers.BatchNormalization(name='bn_3')(shared_x)
    shared_x = layers.Dense(32, name='dense_1')(shared_x)
    shared_x = layers.BatchNormalization(name='bn_4')(shared_x)
    shared_x = layers.LeakyReLU(name='act_2')(shared_x)
    shared_x = layers.Dropout(0.25, name='drop_1')(shared_x)
    
    # head 1 --> reconstruction task
    rot_x = layers.Dense(1, activation='sigmoid', name='rotation')(shared_x)

    # head 2 --> classifier
    cls_x = layers.Dense(4, activation='softmax', name='crop')(shared_x)

    rot_model = tf.keras.Model(inputs=input_x, outputs=[rot_x, cls_x])
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    rot_model.compile(optimizer=opt, loss=[MaskedBXE(scale=1.), MaskedCXE(ignore=4, scale=1.)], 
              metrics=[[MaskedBinaryAccuracyMetric()], [MaskedCategoricalAccuracyMetric(ignore=4)]])
    return rot_model

def get_seg_model():
    tf.keras.backend.clear_session()
    input_x = layers.Input(shape=[10, 7])
    # trunc: shared part of the model
    shared_x = layers.Conv1D(32, 1, name='conv_1')(input_x)
    shared_x = layers.BatchNormalization(name='bn_1')(shared_x)
    shared_x = layers.LeakyReLU(name='act_1')(shared_x)
    shared_x = layers.LSTM(64, dropout=0.25, recurrent_dropout=0.25, return_sequences=True, name='lstm_1')(shared_x)
    shared_x = layers.LSTM(64, return_sequences=False, name='lstm_2')(shared_x)
    shared_x = layers.BatchNormalization(name='bn_2')(shared_x)
    shared_x = layers.Dense(32, name='dense_1')(shared_x)
    shared_x = layers.BatchNormalization(name='bn_3')(shared_x)
    shared_x = layers.LeakyReLU(name='act_2')(shared_x)
    shared_x = layers.Dropout(0.25, name='drop_1')(shared_x)

    # head 1 --> tell the sample time-segment partition.
    seg_x = layers.Dense(5, activation='softmax', name='segment')(shared_x)

    # head 2 --> main classifier for crop
    cls_x = layers.Dense(4, activation='softmax', name='crop')(shared_x)

    seg_model = tf.keras.Model(inputs=input_x, outputs=[seg_x, cls_x])
    seg_model.compile(optimizer='adam', loss=[MaskedCXE(ignore=5), MaskedCXE(ignore=4)], 
          metrics=[[MaskedCategoricalAccuracyMetric(ignore=5)], [MaskedCategoricalAccuracyMetric(ignore=4)]])
    return seg_model

def get_band_model():
    tf.keras.backend.clear_session()
    input_x = layers.Input(shape=[10, 7])
    # trunc: shared part of the model
    shared_x = layers.Conv1D(32, 1, name='conv_1')(input_x)
    shared_x = layers.BatchNormalization(name='bn_1')(shared_x)
    shared_x = layers.LeakyReLU(name='act_1')(shared_x)
    shared_x = layers.LSTM(64, dropout=0.25, recurrent_dropout=0.25, return_sequences=True, name='lstm_1')(shared_x)
    shared_x = layers.LSTM(64, return_sequences=False, name='lstm_2')(shared_x)
    shared_x = layers.BatchNormalization(name='bn_2')(shared_x)
    shared_x = layers.Dense(32, name='dense_1')(shared_x)
    shared_x = layers.BatchNormalization(name='bn_3')(shared_x)
    shared_x = layers.LeakyReLU(name='act_2')(shared_x)
    shared_x = layers.Dropout(0.25, name='drop_1')(shared_x)

    # head 1 --> tell the sample band channel
    ban_x = layers.Dense(7, activation='softmax', name='bands')(shared_x)

    # head 2 --> main classifier for crop
    cls_x = layers.Dense(4, activation='softmax', name='crop')(shared_x)

    band_model = tf.keras.Model(inputs=input_x, outputs=[ban_x, cls_x])
    band_model.compile(optimizer='adam', loss=[MaskedCXE(ignore=7), MaskedCXE(ignore=4)], 
          metrics=[[MaskedCategoricalAccuracyMetric(ignore=7)], [MaskedCategoricalAccuracyMetric(ignore=4)]])
    return band_model

def get_one_for_all_model():
    tf.keras.backend.clear_session()
    input_x = layers.Input(shape=[10, 7])
    # trunc: shared part of the model
    shared_x = layers.Conv1D(32, 1, name='conv_1')(input_x)
    shared_x = layers.BatchNormalization(name='bn_1')(shared_x)
    shared_x = layers.LeakyReLU(name='act_1')(shared_x)
    shared_x = layers.LSTM(64, dropout=0.25, recurrent_dropout=0.25, return_sequences=True, name='lstm_1')(shared_x)
    shared_x = layers.LSTM(64, return_sequences=False, name='lstm_2')(shared_x)
    shared_x = layers.BatchNormalization(name='bn_2')(shared_x)
    shared_x = layers.Dense(32, name='dense_1')(shared_x)
    shared_x = layers.BatchNormalization(name='bn_3')(shared_x)
    shared_x = layers.LeakyReLU(name='act_2')(shared_x)
    shared_x = layers.Dropout(0.25, name='drop_1')(shared_x)

    # head 1 --> tell the sample time-segment partition.
    seg_x = layers.Dense(5, activation='softmax', name='segment')(shared_x)

    # head 2 --> tell the sample band channel
    ban_x = layers.Dense(7, activation='softmax', name='bands')(shared_x)

    # head 3 --> tell the sample rotation
    rot_x = layers.Dense(1, activation='sigmoid', name='rotation')(shared_x)

    # head 4 --> main classifier for crop
    cls_x = layers.Dense(4, activation='softmax', name='crop')(shared_x)

    all_model = tf.keras.Model(inputs=input_x, outputs=[seg_x, ban_x, rot_x, cls_x])
    all_model.compile(optimizer='adam', loss=[MaskedCXE(ignore=5), MaskedCXE(ignore=7), MaskedBXE(), MaskedCXE(ignore=4)], 
              metrics=[[MaskedCategoricalAccuracyMetric(ignore=5)], [MaskedCategoricalAccuracyMetric(ignore=7)], 
                       [MaskedBinaryAccuracyMetric()], [MaskedCategoricalAccuracyMetric(ignore=4)]])
    return all_model


# Test Functions
def test_base_model(y_test, x_test, model=None, name=None):
    test_ds = tf.data.Dataset.from_tensor_slices(x_test).batch(2**14).prefetch(tf.data.experimental.AUTOTUNE)
    y_pred = []
    for x_s in test_ds:
        yhat = model(x_s, training=False)
        y_pred.append(np.argmax(yhat, axis=-1))
            
    y_pred = np.concatenate(y_pred)
    acc = np.sum(y_test == y_pred) / y_test.shape[0]
    
    target_names = ['wheat', 'corn', 'rice', 'other']
    print(classification_report(y_test, y_pred, target_names=target_names))
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    cm_display = ConfusionMatrixDisplay(cm, display_labels=target_names).plot(cmap=plt.cm.Blues)
    plt.grid(False)
    plt.savefig(name)
    plt.show()
    return acc

def test_single_task_model(y_test, x_test, model=None, name=None):
    test_ds = tf.data.Dataset.from_tensor_slices(x_test).batch(2**14).prefetch(tf.data.experimental.AUTOTUNE)
    y_pred = []
    for x_s in test_ds:
        _, yhat = model(x_s, training=False)
        y_pred.append(np.argmax(yhat, axis=-1))
            
    y_pred = np.concatenate(y_pred)
    acc = np.sum(y_test == y_pred) / y_test.shape[0]
    
    target_names = ['wheat', 'corn', 'rice', 'other']
    print(classification_report(y_test, y_pred, target_names=target_names))
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    cm_display = ConfusionMatrixDisplay(cm, display_labels=target_names).plot(cmap=plt.cm.Blues)
    plt.grid(False)
    plt.savefig(name)
    plt.show()
    return acc

def test_multitask_model(y_test, x_test, model=None, name=None):
    test_ds = tf.data.Dataset.from_tensor_slices(x_test).batch(2**14).prefetch(tf.data.experimental.AUTOTUNE)
    y_pred = []
    for x_s in test_ds:
        _, _, _, yhat = model(x_s, training=False)
        y_pred.append(np.argmax(yhat, axis=-1))
            
    y_pred = np.concatenate(y_pred)
    acc = np.sum(y_test == y_pred) / y_test.shape[0]
    
    target_names = ['wheat', 'corn', 'rice', 'other']
    print(classification_report(y_test, y_pred, target_names=target_names))
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    cm_display = ConfusionMatrixDisplay(cm, display_labels=target_names).plot(cmap=plt.cm.Blues)
    plt.grid(False)
    plt.savefig(name)
    plt.show()
    return acc

# Custome Loss and Metrics
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
    def __init__(self, scale=1., **kwargs):
        self.scale = scale
        super().__init__(**kwargs)
        
    def call(self, y_true, y_pred):
        y_true = tf.squeeze(y_true)
        mask = tf.where(y_true!=2)
        y_true = tf.gather_nd(y_true, mask)
        y_pred = tf.gather_nd(y_pred, mask)
        y_true = tf.squeeze(y_true)
        y_pred = tf.squeeze(y_pred)
        loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        return self.scale * tf.reduce_mean(loss)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config}

class MaskedCXE(tf.keras.losses.Loss):
    def __init__(self, ignore=None, scale=1., **kwargs):
        self.ignore = ignore
        self.scale = scale
        super().__init__(**kwargs)
    def call(self, y_true, y_pred):
        y_true = tf.squeeze(y_true)
        mask = tf.where(y_true!=self.ignore)
        y_true = tf.gather_nd(y_true, mask)
        y_pred = tf.gather_nd(y_pred, mask)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        return self.scale * tf.reduce_mean(loss)
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
        return {**base_config, "ignore": self.ignore}
