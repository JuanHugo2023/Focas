from keras.applications import xception
from keras.models import Model, load_model
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, Lambda
from keras.optimizers import RMSprop
from keras.losses import mean_squared_error, poisson
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import backend as K

import numpy as np
import os

from math import ceil

from engine import Engine
from rect import Rect
from image_util import subimage
from point_density import downsample_sum

engine = Engine()

input_shape = (327, 327, 3)  # 327 = 32 * 10 + 7
density_scale = 32
density_shape = (9, 9, 5)
# density_shape = (9, 9, 1)


def build_model():
    # pre-trained model
    base_model = xception.Xception(include_top=False,
                                   weights='imagenet',
                                   input_tensor=None,
                                   input_shape=input_shape,
                                   pooling=None)

    # output from pre-trained model. shape=(?,11,11,2048)
    x = base_model.output

    # ignore the cells closest to the border, shape=(?,9,9,2048)
    x = Lambda(lambda y: y[:, 1:-1, 1:-1, :])(x)

    # add some hidden layers
    for n in [1024, 256]:
        # 1x1 convolution, so assuming we don't need any more information from
        # the neighbouring regions.
        x = Conv2D(n, (1, 1), padding='same', use_bias=False,
                   name='block15_conv{}'.format(n))(x)
        x = BatchNormalization(name='block15_conv{}_bn'.format(n))(x)
        x = Activation('relu', name='block15_conv{}_act'.format(n))(x)

    # final convolution, down to 5 channels (one per kind of sea lion)
    x = Conv2D(density_shape[2], (1, 1), padding='same', use_bias=False,
               name='block15_conv5')(x)
    # softplus ensures our count is not negative, while keeping gradient information 
    x = Activation('softplus', name='block15_conv5_act')(x)

    # disable training for the old layers
    for layer in base_model.layers:
        layer.trainable = False

    model = Model(inputs=base_model.input, outputs=x)
    model
    return model


def random_rect(img_shape, rect_shape):
    img_h, img_w = img_shape[:2]
    rect_h, rect_w = input_shape[:2]
    r = np.random.randint(img_h)
    c = np.random.randint(img_w)
    r0 = r - rect_h // 2
    r1 = r0 + rect_h
    c0 = c - rect_w // 2
    c1 = c0 + rect_w
    return Rect(r0, r1, c0, c1)


def training_density(tid, rect):
    # Final layer convolutional cells are centered at $(32i+3,32j+3)$, for $i,j\geq 0$.
    # So we want the density of a 32x32 square centered at each of these points.
    # These 32x32 pixel squares go outside the input region of the image when $i=0$ or $j=0$
    discard = 19  # number of rows/columns of density not part of an output cell
    rect = Rect(rect.row_min + discard,
                rect.row_max - discard,
                rect.col_min + discard,
                rect.col_max - discard)
    # print(rect.width(), rect.height())
    density = engine.training_density(tid, rect=rect, subsample=2)
    density = density[1:-1, 1:-1, :]
    # print(density.shape)
    density = downsample_sum(density, 64)

    # drop the pup channel, and some the others,
    # giving total non-pup sea lion density
    # density = np.sum(density[:, :, :4], axis=2)[:,:,None]

    assert density.shape == density_shape
    return density


def sample_training_image():
    n_rejected = 0
    while True:
        tid = np.random.randint(len(engine.training_ids()))
        # tid = 0
        tid = engine.training_ids()[tid]
        img = engine.training_mmap_image(tid)
        rect = random_rect(img.shape, input_shape)
        points = engine.training_points(tid, rect=rect, expand=0)
        n = (points[:,0] != 4).sum()
        # print("n={}".format(n))
        p = np.random.rand()
        if p > n:
            n_rejected += 1
            continue
        # print("rejected {} for lack of sea lions".format(n_rejected))

        density = training_density(tid, rect)

        # rect = Rect(1383, 1710, 4734, 5061)
        # rect = Rect(1657,1984,2311,2638)
        # print(tid, (rect.row_min+rect.row_max)//2, (rect.col_min+rect.col_max)//2)
        # print((rect.row_min, rect.row_max, rect.col_min, rect.col_max))
        img = subimage(img, rect)
        mask = engine.training_mmap_mask(tid)
        mask = mask[:,:,None]
        mask = subimage(mask, rect)
        img *= mask
        img = xception.preprocess_input(img.astype(np.float32))
        return img, density


def generate_batches(batch_size):
    while True:
        images, densities = zip(*(sample_training_image()
                                  for i in range(batch_size)))
        images = np.stack(images)
        densities = np.stack(densities)
        yield (images, densities)


def count_error(density_true, density_pred):
    count_true = K.sum(density_true, [1, 2])
    count_pred = K.sum(density_pred, [1, 2])
    diff = count_true - count_pred
    rse = K.sqrt(K.sum(diff ** 2, 1))
    return K.max(rse)


def train_top(model, batch_size, initial_epoch=0, optimizer=None):
    if optimizer is None:
        optimizer = RMSprop()
    model.compile(optimizer=optimizer,
                  loss=mean_squared_error,
                  metrics=[count_error])
    epochs = 1000
    num_images = len(engine.training_ids())
    subimages_per_image = (5616 * 3744) / (input_shape[0] * input_shape[1])
    steps_per_epoch = int(num_images * subimages_per_image / batch_size)
    print("steps_per_epoch={}".format(steps_per_epoch))
    callbacks = [
        ModelCheckpoint('keras_density/weights-more-sealions.{epoch:02d}.hdf5'),
        TensorBoard(log_dir='./keras_density/logs-non-pups/',
                    histogram_freq=1)
    ]
    model.fit_generator(generate_batches(batch_size),
                        epochs=epochs,
                        steps_per_epoch=steps_per_epoch,
                        pickle_safe=True,
                        workers=2,
                        callbacks=callbacks,
                        initial_epoch=initial_epoch)


def predict_large_image(model, img):
    img = xception.preprocess_input(img.astype(np.float32))
    h, w, c = img.shape
    inp_h, inp_w, _ = input_shape
    strides = (8, 8)

    h_ = ceil(h / 32)
    w_ = ceil(w / 32)
    indices = [(i, j) for i in range(-1, h_, strides[0])
                    for j in range(-1, w_, strides[0])]
    print(len(indices), len(indices)*input_shape[0]*input_shape[1]*input_shape[2])

    def get_rect(i,j):
        r0 = i - 19
        r1 = r0 + inp_h
        c0 = j - 19
        c1 = c0 + inp_w
        return Rect(r0, r1, c0, c1)
    rects = [get_rect(i, j) for (i, j) in indices]

    inputs = [subimage(img,rect) for rect in rects]
    inputs = np.stack(inputs, 0)
    print(inputs.shape, inputs.dtype)

    predictions = model.predict(inputs)
    print(predictions.shape, predictions.dtype)

    density_pred = np.zeros((h_, w_, 5), dtype=predictions.dtype)
    print(density_pred.shape, density_pred.dtype)
    cell_counts = np.zeros((h_,w_), dtype=np.int32)
    for (k,((i,j),rect)) in enumerate(zip(indices, rects)):
        i0 = np.clip(i, 0, density_pred.shape[0])
        j0 = np.clip(j, 0, density_pred.shape[1])
        i1 = np.clip(i + density_shape[0], 0, density_pred.shape[0])
        j1 = np.clip(j + density_shape[1], 0, density_pred.shape[1])
        density_pred[i0:i1, j0:j1,:] += predictions[k,(i0-i):(i1-i),(j0-j):(j1-j),:]
        cell_counts[i0:i1, j0:j1] += 1 # ones(density_shape[:2], dtype=cell_counts.dtype)
    assert (cell_counts == 0).sum() == 0
    # density_pred /= cell_counts[:,:,None]
    return density_pred


def training_scores(model, density_pred_path_format=None):
    diff_sq = np.zeros((len(engine.training_ids()), 5))
    for (i, tid) in enumerate(engine.training_ids()):
        img = engine.training_mmap_image(tid)
        path = None
        if density_pred_path_format is not None:
            path = density_pred_path_format.format(tid=tid)
        if path is not None and os.path.isfile(path):
            density_pred = np.load(path)
        else:
            density_pred = predict_large_image(model, img)
            if path is not None and os.path.isfile(path):
                np.save(path.format(tid=tid), density_pred)
        counts_pred = density_pred.sum(axis=(0, 1))
        counts_true = engine.counts.loc[tid, engine.class_names].as_matrix()
        diff_sq[i, :] = (counts_true - counts_pred) ** 2
        print(tid, diff_sq[i, :])
    return np.sqrt(np.mean(diff_sq, axis=0))


if __name__ == '__main__':
    # model = build_model()
    model = load_model('keras_density.hdf5')
    train_top(model, 128)
