from keras.applications import xception
from keras.models import Model, load_model
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, Lambda
from keras.optimizers import RMSprop
from keras.losses import mean_squared_error, poisson
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras import backend as K

import numpy as np
import pandas as pd
import os
import itertools
import multiprocessing

from math import ceil

from engine import Engine
from rect import Rect
from image_util import subimage
from point_density import downsample_sum

engine = Engine()

# input_shape = (327, 327, 3)  # 327 = 32 * 10 + 7
# density_scale = 32
# # density_shape = (9, 9, 5)
# # density_shape = (9, 9, 1)
# density_shape = (7, 7, 5)

# def discard_outer(y):
#     discard = (11-density_shape[0]) // 2
#     return y[:,discard:-discard, discard:-discard, :]

class TrainingRun():
    def __init__(self, loss, optimizer, batch_size, trainable, keep_prob, epochs=None, callbacks=None)
        self.loss = loss
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.trainable = trainable
        self.keep_prob = keep_prob
        self.epochs = epochs
        self.callbacks = callbacks

class Params():
    def __init__(self, name, model_dir, density_dim, density_border, top_layers, predict_batch_size, training_runs=None):
        self.name = name
        self.model_dir = model_dir
        self.density_dim = density_dim
        self.density_border = density_border
        self.top_layers = top_layers
        self.cell_size = 32  # determined by base trained net
        self.border = 19  # determined by base trained net
        self.num_classes = 5  # determined by problem
        self.density_shape = (self.density_dim,
                              self.density_dim,
                              self.num_classes)
        self.input_shape = (self.density_dim + 2 * self.density_border) * self.cell_size + 2 * self.BORDER
        self.predict_batch_size = predict_batch_size
        self.training_runs = training_runs
        self.training_run_index = 0

    def training_run():
        return self.training_runs[self.training_run_index]

    def weights_path(run):



def conv_layer(filters, kernel_size, padding='same', name=None, normalization=True, activation=True):
    def _fn(x):
        x = Conv2D(filters, kernel_size,
                   padding=padding,
                   use_bias=False,
                   name=name)(x)
        if normalization:
            x = BatchNormalization(name=name+'_bn')(x)
        if activation:
            x = Activation('relu', name=name+'_act')(x)
        return x

def final_conv_layer(filters, kernel_size, padding='same', name=None):
    return conv_layer(filters, kerenl_size, padding=padding, name=name, normalization=False, activation=False)

def build_model(params):

    # pre-trained model
    base_model = xception.Xception(include_top=False,
                                   weights='imagenet',
                                   input_tensor=None,
                                   input_shape=input_shape,
                                   pooling=None)

    # disable training for the pre-trained layers
    for layer in base_model.layers:
        layer.trainable = False

    # output from pre-trained model. shape=(?,11,11,2048)
    x = base_model.output

    for layer in top_layers:
        x = layer(x)

    # final convolution, down to 5 channels (one per kind of sea lion)
    # x = Conv2D(params.num_classes, (1, 1), padding='same', use_bias=False,
    #            name='block15_conv5')(x)
    # softplus ensures our count is not negative, while keeping gradient information 
    # x = Activation('softplus', name='block15_conv5_act')(x)

    d = x.shape[1] - params.density_dim
    assert d >= 0, "output is smaller than specified density_dim"
    if d > 0:
        x = x[:, d:-d, d:-d]

    model = Model(inputs=base_model.input, outputs=x)
    return model

# model = build_model(7, 1, [conv_layer(1024, (3,3)), conv_layer(256, (1,1))])


def random_rect(img_shape, rect_shape):
    img_h, img_w = img_shape[:2]
    rect_h, rect_w = rect_shape[:2]
    r = np.random.randint(img_h)
    c = np.random.randint(img_w)
    r0 = r - rect_h // 2
    r1 = r0 + rect_h
    c0 = c - rect_w // 2
    c1 = c0 + rect_w
    return Rect(r0, r1, c0, c1)


def training_density(tid, rect, params):
    # Final layer convolutional cells are centered at $(32i+3,32j+3)$, for $i,j\geq 0$.
    # So we want the density of a 32x32 square centered at each of these points.
    # These 32x32 pixel squares go outside the input region of the image when $i=0$ or $j=0$

    # number of rows/columns of density not part of an output cell
    discard = params.density_border * params.cell_size + params.border
    rect = Rect(rect.row_min + discard,
                rect.row_max - discard,
                rect.col_min + discard,
                rect.col_max - discard)
    density = engine.training_density(tid, rect=rect, subsample=2)
    density = density[1:-1, 1:-1, :]
    density = downsample_sum(density, 64)

    assert density.shape == (params.density_dim, params.density_dim, params.num_classes)
    return density

def reject_empty(p_empty):
    def fn(tid, rect):
        points = engine.training_points(tid, rect=rect)
        if points.shape[0] == 0:
            return p_empty
        else:
            return 1.0
    return fn

def sample_orientation():
    bits = [np.random.rand() > 0.5 for i in range(3)]

    def apply_orientation(img):
        for axis in range(2):
            if bits[axis]:
                img = np.flip(img, axis)
        if bits[2]:
            # transpose
            img = np.swapaxes(img, 0, 1)
        return img

    return apply_orientation

def sample_training_image(params):
    n_rejected = 0
    while True:
        tid = np.random.randint(len(engine.training_ids()))
        tid = engine.training_ids()[tid]
        img = engine.training_mmap_image(tid)
        rect = random_rect(img.shape, (params.input_dim, params.input_dim))
        if np.random.rand() > params.keep_prob(tid, rect):
            continue

        density = training_density(tid, rect, params)

        img = subimage(img, rect)
        mask = engine.training_mmap_mask(tid)
        mask = mask[:,:,None]
        mask = subimage(mask, rect)
        img *= mask
        img = xception.preprocess_input(img.astype(np.float32))
        orient = sample_orientation()
        img = orient(img)
        density = orient(density)
        return img, density


def generate_batches(params):
    while True:
        images, densities = zip(*(sample_training_image(params)
                                  for i in range(params.batch_size)))
        images = np.stack(images)
        densities = np.stack(densities)
        yield (images, densities)


def count_error(density_true, density_pred):
    count_true = K.sum(density_true, [1, 2])
    count_pred = K.sum(density_pred, [1, 2])
    diff = count_true - count_pred
    diff = K.abs(diff)
    print("diff abs", diff.shape)
    diff = K.mean(diff, axis=1)
    print("diff mean", diff.shape)
    return K.max(diff, axis=0)


def area_loss(density_true, density_pred):
    print("density_true.shape",density_true.shape)
    def sums(density, n):
        return K.pool2d(density_pred, (n, n), pool_mode='avg') * n * n
    def rmse_per_channel(_density_true, _density_pred):
        diff_sq = (_density_true - _density_pred) ** 2
        mse = K.mean(diff_sq, axis=[1, 2])
        return mse
        # return K.sqrt(mse)
    loss = rmse_per_channel(density_true, density_pred)
    print("loss per channel",loss.shape)

    for n in range(3, density_shape[0]+1, 2):
        sums_true = sums(density_true, n)
        sums_pred = sums(density_pred, n)
        assert sums_true.shape[1:] == (density_shape[0] - n + 1, density_shape[1] - n + 1, density_shape[2])
        loss += rmse_per_channel(sums_true, sums_pred)

    loss = K.mean(loss, axis=1)
    print("loss",loss.shape)
    return loss


def train_top(model, batch_size, initial_epoch=0, optimizer=None,
              loss=mean_squared_error,
              run_name=''):
    if optimizer is None:
        optimizer = RMSprop()
    model.compile(optimizer=optimizer,
                    loss=loss,
                    metrics=[count_error])
    epochs = 1000
    num_images = len(engine.training_ids())
    subimages_per_image = (5616 * 3744) / (input_shape[0] * input_shape[1])
    steps_per_epoch = int(num_images * subimages_per_image / batch_size)
    print("steps_per_epoch={}".format(steps_per_epoch))
    checkpoint_format = 'keras_density/weights-{}.{{epoch:02d}}.hdf5'.format(run_name)
    log_dir = './keras_density/logs-{}/'.format(run_name)
    callbacks = [
        ModelCheckpoint(checkpoint_format),
        TensorBoard(log_dir=log_dir,
                    histogram_freq=1),
        ReduceLROnPlateau(patience=1)
    ]
    model.fit_generator(generate_batches(batch_size),
                        epochs=epochs,
                        steps_per_epoch=steps_per_epoch,
                        pickle_safe=True,
                        workers=2,
                        callbacks=callbacks,
                        initial_epoch=initial_epoch)


def batch_inputs(inputs, batch_size=64):
    batch = []
    for inp in inputs:
        batch.append(inp)
        if len(batch) >= batch_size:
            yield np.stack(batch, axis=0)
            batch = []
    if len(batch) > 0:
        yield np.stack(batch, axis=0)

def predict_large_image(model, img):
    img = xception.preprocess_input(img.astype(np.float32))
    h, w, c = img.shape
    inp_h, inp_w, _ = input_shape
    strides = (7, 7)

    h_ = ceil(h / 32)
    w_ = ceil(w / 32)
    indices_ = [(i, j) for i in range(-1, h_, strides[0])
                for j in range(-1, w_, strides[0])]
    indices = [(i*32, j*32) for (i, j) in indices_]
    # print(len(indices), len(indices)*input_shape[0]*input_shape[1]*input_shape[2])

    def get_rect(i,j):
        r0 = i - 19
        r1 = r0 + inp_h
        c0 = j - 19
        c1 = c0 + inp_w
        return Rect(r0, r1, c0, c1)
    rects = [get_rect(i, j) for (i, j) in indices]

    inputs = (subimage(img, rect) for rect in rects)

    batch_size = 78
    steps = ceil(len(indices) / batch_size)
    inputs = batch_inputs(inputs, batch_size)
    zero_inputs = itertools.repeat(np.zeros((inp_h, inp_w, 3)))
    inputs = itertools.chain(inputs, zero_inputs)

    predictions = model.predict_generator(inputs, steps)

    # inputs = np.stack(inputs, 0)
    # print(inputs.shape, inputs.dtype)

    # predictions = model.predict(inputs)
    # print(predictions.shape, predictions.dtype)

    density_pred = np.zeros((h_, w_, 5), dtype=predictions.dtype)
    # print(density_pred.shape, density_pred.dtype)
    cell_counts = np.zeros((h_,w_), dtype=np.int32)
    for (k, (i, j)) in enumerate(indices_):
        i0 = np.clip(i, 0, density_pred.shape[0])
        j0 = np.clip(j, 0, density_pred.shape[1])
        i1 = np.clip(i + density_shape[0], 0, density_pred.shape[0])
        j1 = np.clip(j + density_shape[1], 0, density_pred.shape[1])
        # i0 = np.clip(i+1, 0, density_pred.shape[0])
        # j0 = np.clip(j+1, 0, density_pred.shape[1])
        # i1 = np.clip(i + density_shape[0] - 1, 0, density_pred.shape[0])
        # j1 = np.clip(j + density_shape[1] - 1, 0, density_pred.shape[1])
        density_pred[i0:i1, j0:j1, :] += predictions[k, (i0-i):(i1-i),
                                                     (j0-j):(j1-j), :]
        cell_counts[i0:i1, j0:j1] += 1
    assert (cell_counts == 0).sum() == 0
    density_pred /= cell_counts[:, :, None]
    return density_pred


def training_scores(model):
    diff = np.zeros((len(engine.training_ids()), 5))
    diff_ratio = np.zeros((len(engine.training_ids()), 5))
    training_ids = engine.training_ids().copy()
    np.random.shuffle(training_ids)
    for (i, tid) in enumerate(training_ids):
        print("tid:{} ({}/{})".format(tid, i, len(training_ids)))
        img = engine.training_mmap_image(tid)
        mask = engine.training_mmap_mask(tid)
        img = img*mask[:,:,None]
        density_pred = predict_large_image(model, img)
        counts_pred = density_pred.sum(axis=(0, 1))
        counts_true = engine.counts.loc[tid, engine.class_names].as_matrix()
        diff[i, :] = counts_pred - counts_true
        avg_so_far = np.mean(diff[:i+1, :] ** 2, axis=0)
        diff_ratio[i, :] = counts_true / counts_pred
        print("counts_true:", counts_true)
        print("counts_pred:", counts_pred)
        print("diff:", diff[i, :])
        print("diff_ratio", diff_ratio[i, :])
        print("avg_diff_ratio", np.mean(diff_ratio[:i+1, :], axis=0))
        print("diff_sq:", diff[i, :] ** 2)
        print("avg_diff_sq:", avg_so_far)
        print("score:", np.mean(np.sqrt(diff[i, :] ** 2)))
        print("cumm score:", np.mean(np.sqrt(avg_so_far)))
        print("")
    return training_ids, diff

def load_test_image(test_id):
    global _test_image_queue
    queue = _test_image_queue
    img = engine.test_image(test_id)
    queue.put((test_id, img))

def predict_test_images(model, results_path):
    test_counts = engine.test_counts(results_path)
    test_ids = engine.test_ids()
    total_tests = len(test_ids)
    test_ids = [tid for tid in test_ids if tid not in test_counts.index]
    test_ids.sort()
    remaining_tests = len(test_ids)

    queue = multiprocessing.Queue(2)
    global _test_image_queue
    _test_image_queue = queue

    with multiprocessing.Pool(2) as pool:
        pool.map_async(load_test_image, test_ids)
        with open(results_path, 'a') as f:
            while remaining_tests > 0:
                test_id, img = queue.get()
                print("test image {} ({}/{})".format(test_id, total_tests - remaining_tests, total_tests))
                remaining_tests -= 1
                density_pred = predict_large_image(model, img)
                counts_pred = density_pred.sum(axis=(0, 1))
                f.write("{},{},{},{},{},{}\n".format(
                    test_id, counts_pred[0], counts_pred[1], counts_pred[2], counts_pred[3], counts_pred[4]))
                f.flush()

def params_alpha():
    # training score 14.?
    return Params(name='alpha',
                  model_dir='model_data/'
                  density_dim=7,
                  density_border=1,
                  top_layers=[
                      conv_layer(1024, (3,3), "block15_conv1024"),
                      conv_layer(256, (1,1), "block15_conv256"),
                      final_conv_layer(5, (1, 1), "block15_conv5"),
                      Activation('softplus', name='block15_conv5_act')],
                  predict_batch_size=64,
                  training_runs=[
                      TrainingRun(loss = area_loss,
                                  optimizer = RMSprop(1e-3),
                                  batch_size = 64,
                                  trainable = lambda layer: str.startswith(layer.name, "block15"),
                                  keep_prob = reject_empty(0.05),
                                  callbacks=[EarlyStopping(?)]),
                      TrainingRun(loss = area_loss,
                                  optimizer = RMSprop(1e-5),
                                  batch_size = 64,
                                  trainable = lambda layer: (str.startswith(layer.name, "block14") ||
                                                             str.startswith(layer.name, "block15")),
                                  keep_prob = reject_empty(1.0),
                                  callbacks=[ReduceLROnPlateau(patience=0), EarlyStopping(?)])
                  ])

if __name__ == '__main__':
    model = build_model(params_alpha())
    # model.load_weights('keras_density.hdf5')
    # model = load_model('keras_density.hdf5')
    # train_top(model, 128)
