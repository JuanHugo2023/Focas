from keras.applications import xception
from keras.models import Model, load_model
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, Lambda
from keras.optimizers import RMSprop
from keras.losses import mean_squared_error, poisson
from keras.callbacks import ModelCheckpoint, TensorBoard, \
    ReduceLROnPlateau, EarlyStopping
from keras.engine.training import GeneratorEnqueuer
from keras import backend as K

import numpy as np
import pandas as pd
import os
import time
import itertools
import multiprocessing
import importlib

from math import ceil

from rect import Rect
from image_util import subimage
from point_density import downsample_sum

import engine as engine_lib


epoch = 0
training_run = 0
params = None
checkpoint_format = '{model_dir}/weights-{run:02d}-{{epoch:02d}}.hdf5'
final_weights_format = '{model_dir}/weights-{run:02d}.hdf5'

def checkpoint_path(_training_run=None, _epoch=None):
    """Return the path to a checkpoint file.

    Args:
    _training_run (int, optional): the training run index.
        Defaults to current training run.
    _epoch (int, optional): the epoch.
        Defaults to the current epoch
    """
    if _training_run is None:
        _training_run = training_run
    if _epoch is None:
        _epoch = epoch
    path = checkpoint_format.format(model_dir=params.model_dir,
                                    run=_training_run)\
                            .format(epoch=_epoch)
    return path

def final_weights_path(_training_run=None):
    """Return the path to the weights at the end of a training run.

    Args:
    _training_run (int, optional): the training run index.
        Defaults to current training run.
    """
    if _training_run is None:
        _training_run = training_run
    path = final_weights_format.format(model_dir=params.model_dir,
                                       run=_training_run)
    return path

def find_training_status():
    """Find the current training run/epoch.

    This is based on which files are present in the model_dir
    """
    global training_run
    global epoch
    training_run = 0
    epoch = None
    while os.path.isfile(final_weights_path()):
        training_run += 1
    epoch = 0
    while os.path.isfile(checkpoint_path()):
        epoch += 1


def build_model():
    """Build the Keras model
    """
    # pre-trained model
    base_model = xception.Xception(include_top=False,
                                   weights='imagenet',
                                   input_tensor=None,
                                   input_shape=params.input_shape,
                                   pooling=None)

    # disable training for the pre-trained layers
    for layer in base_model.layers:
        layer.trainable = False

    # output from pre-trained model. shape=(?,11,11,2048)
    x = base_model.output

    for layer in params.top_layers:
        x = layer(x)

    d = int(x.shape[1] - params.density_dim) // 2
    assert d >= 0, "output is smaller than specified density_dim"
    if d > 0:
        x = Lambda(lambda y: y[:, d:-d, d:-d])(x)

    model = Model(inputs=base_model.input, outputs=x)
    if epoch <= 0:
        path = final_weights_path(training_run-1)
    else:
        path = checkpoint_path(training_run, epoch-1)
    if os.path.isfile(path):
        print("loading weights from {}".format(path))
        model.load_weights(path)
    else:
        print("no weights file found")
    return model


def random_rect(img_shape, rect_shape):
    """Returns a random rect, mostly inside the image.

    The algorithm is to pick a random centre point inside the image,
    so up to half the rows/cols might be outside the image.
    """
    img_h, img_w = img_shape[:2]
    rect_h, rect_w = rect_shape[:2]
    r = np.random.randint(img_h)
    c = np.random.randint(img_w)
    r0 = r - rect_h // 2
    r1 = r0 + rect_h
    c0 = c - rect_w // 2
    c1 = c0 + rect_w
    return Rect(r0, r1, c0, c1)


def training_density(tid, rect):
    """Return the true density of a piece of a training image
    Args:
    tid (int): training image id
    """
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

    assert density.shape == (params.density_dim,
                             params.density_dim,
                             params.num_classes)
    return density


def sample_orientation():
    """Returns a function which randomly flips/transposes an image.

    The returned function performs the SAME flips/transpositions each time it
    is called, so you can apply the same transformations to both the input
    image and the target densities.
    """
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


def sample_training_image(run):
    """Return a random piece of a random training image, ready to feed into NN.
    """
    while True:
        tid = np.random.choice(engine.training_ids)
        img = engine.training_mmap_image(tid)
        rect = random_rect(img.shape, (params.input_dim, params.input_dim))
        if np.random.rand() > run.keep_prob(tid, rect):
            continue

        density = training_density(tid, rect)

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


def do_training_run(model):
    """Do a training run.
    """
    global training_run
    global epoch
    if training_run >= len(params.training_runs):
        return False
    print("beginning training run {}/{}".format(training_run+1, len(params.training_runs)))
    run = params.training_runs[training_run]
    for layer in model.layers:
        layer.trainable = run.trainable(layer)
    batch_size = run.batch_size
    optimizer = run.optimizer
    if optimizer is None:
        optimizer = RMSprop()
    loss = run.loss
    metrics = run.metrics
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)
    epochs = 1000

    # Compute approximate size of epochs.
    # We don't really have epochs since we are just sampling each training
    # case independently. This is actually approximately the number of
    # training cases needed to cover the training image set, assuming we throw
    # out the non-sea lion parts of the images.
    num_images = len(engine.training_ids)
    image_size = 5616 * 3744
    input_size = params.input_shape[0] * params.input_shape[1]
    subimages_per_image = image_size / input_size
    subimages_per_image /= 20  # about 1/20 squares have sea lions
    steps_per_epoch = int(num_images * subimages_per_image / batch_size)

    _checkpoint_format = checkpoint_format.format(model_dir=params.model_dir,
                                                  run=training_run)
    log_dir = '{}/logs/'.format(params.model_dir)
    callbacks = run.callbacks + [
        ModelCheckpoint(_checkpoint_format),
        TensorBoard(log_dir=log_dir,
                    histogram_freq=1)]
    generator = batch_inputs(sample_training_image(run) for i in itertools.count())
    model.fit_generator(generator,
                        epochs=epochs,
                        steps_per_epoch=steps_per_epoch,
                        pickle_safe=True,
                        workers=2,
                        callbacks=callbacks,
                        initial_epoch=epoch)
    model.save_weights(final_weights_path())
    training_run += 1
    epoch = 0
    return True


def batch_inputs(inputs, batch_size=64):
    """Batches together values from an input iterator.

    The batches are stacked together into numpy arrays.
    If inputs generates numpy arrays, this yields a numpy array per batch.
    If inputs generates tuples, the result is a tuple of numpy arrays per batch.

    Args:
    inputs (iterator): Should produce either arrays or tuples of arrays.
    batch_size (int): Number of inputs to batch together for each output.
    """
    batch = []
    for inp in inputs:
        batch.append(inp)
        if len(batch) >= batch_size:
            if isinstance(batch[0], tuple):
                inputs = zip(*batch)
                stacked = tuple(np.stack(inp, axis=0) for inp in inputs)
            else:
                stacked = np.stack(batch, axis=0)
            yield stacked
            batch = []
    if len(batch) > 0:
        yield np.stack(batch, axis=0)


def predict_large_image(model, img):
    """Predicts the density of a full image.
    """
    img = xception.preprocess_input(img.astype(np.float32))
    h, w, c = img.shape
    inp_h, inp_w, _ = params.input_shape

    h_ = ceil(h / params.cell_size)
    w_ = ceil(w / params.cell_size)
    indices_ = [(i, j) for i in range(-1, h_, params.density_dim)
                for j in range(-1, w_, params.density_dim)]
    indices = [(i*params.cell_size, j*params.cell_size) for (i, j) in indices_]

    def get_rect(i, j):
        r0 = i - params.border
        r1 = r0 + inp_h
        c0 = j - params.border
        c1 = c0 + inp_w
        return Rect(r0, r1, c0, c1)
    rects = [get_rect(i, j) for (i, j) in indices]

    inputs = (subimage(img, rect) for rect in rects)

    batch_size = params.predict_batch_size
    steps = ceil(len(indices) / batch_size)
    inputs = batch_inputs(inputs, batch_size)
    zero_inputs = itertools.repeat(np.zeros((inp_h, inp_w, 3)))
    inputs = itertools.chain(inputs, zero_inputs)

    predictions = model.predict_generator(inputs, steps)

    density_pred = np.zeros((h_, w_, params.num_classes),
                            dtype=predictions.dtype)
    cell_counts = np.zeros((h_,w_), dtype=np.int32)
    for (k, (i, j)) in enumerate(indices_):
        i0 = np.clip(i, 0, density_pred.shape[0])
        j0 = np.clip(j, 0, density_pred.shape[1])
        i1 = np.clip(i + params.density_dim, 0, density_pred.shape[0])
        j1 = np.clip(j + params.density_dim, 0, density_pred.shape[1])
        # i0 = np.clip(i + params.density_border,
        #              0, density_pred.shape[0])
        # j0 = np.clip(j + params.density_border,
        #              0, density_pred.shape[1])
        # i1 = np.clip(i + params.density_dim - params.density_border,
        #              0, density_pred.shape[0])
        # j1 = np.clip(j + params.density_dim - params.density_border,
        #              0, density_pred.shape[1])
        density_pred[i0:i1, j0:j1, :] += predictions[k, (i0-i):(i1-i),
                                                     (j0-j):(j1-j), :]
        cell_counts[i0:i1, j0:j1] += 1
    assert (cell_counts == 0).sum() == 0
    density_pred /= cell_counts[:, :, None]
    return density_pred


class AvgStopWatch():
    def __init__(self):
        self.t0 = time.time()
        self.running_avg = None

    def lap(self):
        t = time.time()
        dt = t - self.t0
        self.t0 = t
        if self.running_avg is None:
            self.running_avg = dt
        else:
            self.running_avg = 0.9 * self.running_avg + 0.1 * dt
        return self.running_avg


def generate_count_predictions(model, tids, get_img,
                               pickle_safe=True, results_path=None):
    """Predict the counts for a sequence of images.

    Args:
    model: model for prediction
    tids (iterable): ids of images to predict counts for
    get_img (id -> Image): function which takes an id in tids and returns an image.
    pickle_safe (bool): is it safe to get the other images in another process?
        Defaults to True.
    results_path (str, optional): path to save counts. If specified, counts
        are appended as CSV rows after each image, and any images that already
        have a row in this file are skipped.
    """
    num_images = len(tids)

    if results_path is not None and os.path.isfile(results_path):
        saved_counts = engine.load_counts(results_path)
        tids = [tid for tid in tids
                if tid not in saved_counts.index]
        tids.sort()

    remaining_images = len(tids)

    generator = ((tid, get_img(tid)) for tid in tids)
    enqueuer = GeneratorEnqueuer(generator, pickle_safe)
    enqueuer.start(max_q_size=2)
    wait_time = 0.01

    timer = AvgStopWatch()

    while True:
        generator_output = None
        while enqueuer.is_running():
            if not enqueuer.queue.empty():
                generator_output = enqueuer.queue.get()
                break
            else:
                time.sleep(wait_time)

        if generator_output is None:
            break
        tid, img = generator_output
        density = predict_large_image(model, img)
        counts = density.sum(axis=(0, 1))

        dt = timer.lap()
        remaining_time = dt * remaining_images
        remaining_images -= 1

        print("image {} ({}/{}) -- {} s remaining".format(tid,
                                                          num_images - remaining_images,
                                                          num_images,
                                                          remaining_time))

        if results_path is not None:
            with open(results_path, 'a') as f:
                f.write("{},{},{},{},{},{}\n".format(
                    tid, counts[0], counts[1], counts[2], counts[3], counts[4]))
                f.flush()
        yield tid, counts


def train(args):
    """Entry point for training"""
    model = args.model
    while do_training_run(model):
        pass


def validate(args):
    """Entry point for validation"""
    model = args.model

    validation_ids = engine.validation_ids
    diff = np.zeros((len(validation_ids), 5))
    diff_ratio = np.zeros((len(validation_ids), 5))

    predictions = generate_count_predictions(model,
                                             validation_ids,
                                             engine.training_image,
                                             pickle_safe=True,
                                             results_path=args.results_path)

    for i, (tid, counts) in enumerate(predictions):
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
        print("score:", np.mean(np.sqrt(diff[i, :] ** 2)))
        print("cumm score:", np.mean(np.sqrt(avg_so_far)))
        print("")

    df = pd.DataFrame(diff, index=validation_ids, columns=engine.class_names)
    df.index.name = 'train_id'
    return df


def test(args):
    """Entry point for predicting the test images"""
    model = args.model

    predictions = generate_count_predictions(model,
                                             engine.test_ids(),
                                             engine.test_image,
                                             pickle_safe=True,
                                             results_path=args.results_path)
    for x in predictions:
        pass


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model')

    subparsers = parser.add_subparsers(help='sub-command help',
                                       dest='command')
    parser_train = subparsers.add_parser('train', help='train help')
    parser_train.set_defaults(func=train)

    parser_valid = subparsers.add_parser('validate', help='valid help')
    parser_valid.add_argument('--output', '-o', dest='results_path', required=True)
    parser_valid.set_defaults(func=validate)

    parser_test = subparsers.add_parser('test', help='test help')
    parser_test.add_argument('--output', '-o', dest='results_path', required=True)
    parser_test.set_defaults(func=test)

    args = parser.parse_args()

    global engine
    engine_lib.init_engine()
    engine = engine_lib.get_engine()

    global params
    params_mod = importlib.import_module('density_'+args.model)
    params = params_mod.params

    try:
        os.makedirs(params.model_dir)
    except:
        pass

    find_training_status()

    model = build_model()

    args.model = model
    args.func(args)

if __name__ == '__main__':
    main()
