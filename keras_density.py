from keras.applications import xception
from keras.models import Model
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, Lambda
from keras.optimizers import RMSprop
from keras.losses import mean_squared_error
from keras.callbacks import ModelCheckpoint

import numpy as np

from engine import Engine
from rect import Rect
from image_util import subimage
from point_density import downsample_sum

engine = Engine()

input_shape = (327, 327, 3)  # 327 = 32 * 10 + 7
density_scale = 32
density_shape = (9, 9, 5)


def build_model():
    # pre-trained model
    base_model = xception.Xception(include_top=False,
                                   weights='imagenet',
                                   input_tensor=None,
                                   input_shape=input_shape,
                                   pooling=None)

    # output from pre-trained model. shape=(?,10,10,2048)
    x = base_model.output

    # add some top layers.
    for n in [256, 5]:
        # 1x1 convolution, so assuming we don't need any more information from
        # the neighbouring regions.
        x = Conv2D(n, (1, 1), padding='same', use_bias=False,
                   name='block15_conv{}'.format(n))(x)
        x = BatchNormalization(name='block15_conv{}_bn'.format(n))(x)
        x = Activation('relu', name='block15_conv{}_act'.format(n))(x)
    x = Lambda(lambda y: y[:, 1:-1, 1:-1, :])(x)

    # disable training for the old layers
    for layer in base_model.layers:
        layer.trainable = False

    model = Model(inputs=base_model.input, outputs=x)
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
    # So we want the density of a 32x32 pixel square centered at each of these points.
    # These 32x32 pixel squares go outside the input region of the image when $i=0$ or $j=0$
    # centres of convolution cell in final layer are at
    discard = 19  # number of rows/columns of density not part of an output cell
    rect = Rect(rect.row_min + discard,
                rect.row_max - discard,
                rect.col_min + discard,
                rect.col_max - discard)
    # print(rect.width(), rect.height())
    density = engine.training_density(tid, rect=rect, subsample=2)
    density = density[1:-1, 1:-1]
    density = downsample_sum(density, 64)
    assert density.shape == density_shape
    return density


def sample_training_image():
    tid = np.random.randint(len(engine.training_ids()))
    # tid = 0
    tid = engine.training_ids()[tid]
    img = engine.training_mmap_image(tid)
    mask = engine.training_mmap_mask(tid)
    mask = mask[:,:,None]
    rect = random_rect(img.shape, input_shape)
    #rect = Rect(1383, 1710, 4734, 5061)
    # rect = Rect(1657,1984,2311,2638)
    # print(tid, (rect.row_min+rect.row_max)//2, (rect.col_min+rect.col_max)//2)
    # print((rect.row_min, rect.row_max, rect.col_min, rect.col_max))
    img = subimage(img, rect)
    mask = subimage(mask, rect)
    img *= mask
    img = xception.preprocess_input(img.astype(np.float32))
    density = None
    density = training_density(tid, rect)
    return img, density


def generate_batches(batch_size):
    while True:
        images, densities = zip(*(sample_training_image()
                                  for i in range(batch_size)))
        images = np.stack(images)
        densities = np.stack(densities)
        yield (images, densities)

def train_top(model, batch_size):
    model.compile(optimizer=RMSprop(), loss=mean_squared_error)
    num_images = len(engine.training_ids())
    subimages_per_image = (5616 * 3744) / (input_shape[0] * input_shape[1])
    steps_per_epoch = int(num_images * subimages_per_image / batch_size)
    print("steps_per_epoch={}".format(steps_per_epoch))
    callbacks = [
        ModelCheckpoint('keras_density/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                        save_weights_only=True)
    ]
    model.fit_generator(generate_batches(batch_size),
                        steps_per_epoch=steps_per_epoch,
                        pickle_safe=True,
                        callbacks=callbacks)

if __name__ == '__main__':
    model = build_model()
    train_top(model, 128)
