from keras.applications import xception
from keras.models import Model, load_model
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, Lambda
from keras.optimizers import RMSprop
from keras.losses import mean_squared_error, poisson
from keras.callbacks import ModelCheckpoint, TensorBoard, \
    ReduceLROnPlateau, EarlyStopping
from keras import backend as K

import engine as engine_lib


#  HELPER FUNCTIONS

def conv_layer(filters,
               kernel_size,
               padding='same',
               name=None,
               normalization=True,
               activation=True):
    def _fn(x):
        x = Conv2D(
            filters, kernel_size, padding=padding, use_bias=False,
            name=name)(x)
        if normalization:
            x = BatchNormalization(name=name + '_bn')(x)
        if activation:
            x = Activation('relu', name=name + '_act')(x)
        return x
    return _fn


def final_conv_layer(filters, kernel_size, padding='same', name=None):
    return conv_layer(
        filters,
        kernel_size,
        padding=padding,
        name=name,
        normalization=False,
        activation=False)


def count_error(density_true, density_pred):
    count_true = K.sum(density_true, [1, 2])
    count_pred = K.sum(density_pred, [1, 2])
    diff = count_true - count_pred
    diff = K.abs(diff)
    diff = K.mean(diff, axis=1)
    return K.max(diff, axis=0)


def reject_empty(p_empty):
    def fn(tid, rect):
        engine = engine_lib.get_engine()
        points = engine.training_points(tid, rect=rect)
        if points.shape[0] == 0:
            return p_empty
        else:
            return 1.0
    return fn

class TrainingRun():
    def __init__(self, loss, optimizer, batch_size, trainable, keep_prob,
                 epochs=None, callbacks=None, metrics=None):
        self.loss = loss
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.trainable = trainable
        self.keep_prob = keep_prob
        self.epochs = epochs
        self.callbacks = callbacks
        self.metrics = metrics

class Params():
    # def __init__(self):
    #     for name in self.param_names + self.const_names:
    #         value = getattr(self, name, None)
    #         if value is None:
    #             fn = getattr(self, '_'+name, None)
    #             if fn is not None:
    #                 value = fn()
    #         setattr(self, name, value)
    # param_names = ['model_name',
    #                'model_dir',
    #                'density_dim',
    #                'density_border',
    #                'top_layers',
    #                'predict_batch_size',
    #                'training_runs']
    # const_names = ['cell_size',
    #                'border',
    #                'num_classes',
    #                'density_shape',
    #                'input_dim',
    #                'input_shape']

    cell_size = 32

    border = 19

    num_classes = 5

    @property
    def density_shape(self):
        return (self.density_dim,
                self.density_dim,
                self.num_classes)

    @property
    def input_dim(self):
        return (self.density_dim + 2 * self.density_border) * self.cell_size + 2 * self.border + 1

    @property
    def input_shape(self):
        return (self.input_dim, self.input_dim, 3)

    # def init_constants(self):
    #     # INITIALIZE CONSTANTS
    #     self.cell_size = 32  # determined by base trained net
    #     self.border = 19  # determined by base trained net
    #     self.num_classes = 5  # determined by problem

    #     self.density_shape = (self.density_dim,
    #                           self.density_dim,
    #                           self.num_classes)
    #     self.input_dim = (self.density_dim + 2 * self.density_border) * self.cell_size + 2 * self.border + 1
    #     self.input_shape = (self.input_dim, self.input_dim, 3)

    # DEFAULT PARAMS

    model_name = 'default'

    model_dir = 'model_data/default/'

    density_dim = 1

    density_border = 0

    top_layers = []

    predict_batch_size = 64

    training_runs = []

    def area_loss(self, density_true, density_pred):
        def sums(density, n):
            return K.pool2d(density_pred, (n, n), pool_mode='avg') * n * n
        def rmse_per_channel(_density_true, _density_pred):
            diff_sq = (_density_true - _density_pred) ** 2
            mse = K.mean(diff_sq, axis=[1, 2])
            return mse
        loss = rmse_per_channel(density_true, density_pred)

        for n in range(3, self.density_dim+1, 2):
            sums_true = sums(density_true, n)
            sums_pred = sums(density_pred, n)
            loss += rmse_per_channel(sums_true, sums_pred)

        loss = K.mean(loss, axis=1)
        return loss
