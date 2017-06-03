
import numpy as np
import tensorflow as tf
import os
import sys
# from pathos.multiprocessing import ProcessingPool
from multiprocessing import Process, Queue
from functools import wraps

from darkflow.net import framework
from darkflow.net.build import TFNet
from darkflow.defaults import argHandler

from engine import Engine
from image_util import subimage, random_subimage_rect
from rect import Rect

def _process_iter(q, f, args, kwargs):
    print("_process_iter")
    for x in f(*args, **kwargs):
        q.put((x, False))
    q.put((None, True))
    q.close()

def process_iter(queue_size):
    def get_f(f):
        @wraps(f)
        def get_args(*args, **kwargs):
            q = Queue(queue_size)
            p = Process(target=_process_iter, args=(q, f, args, kwargs))
            p.start()
            while True:
                value, closed = q.get()
                if closed:
                    break
                yield value
        return get_args
    return get_f


class CountFramework(framework.framework):
    def constructor(self, meta, FLAGS):
        self.engine = Engine()
        self.meta = meta
        self.FLAGS = FLAGS
        self.fetch = []

    def parse(self, exclusive=False):
        pass

    # @process_iter(2)
    def shuffle(self):
        batch = self.FLAGS.batch
        size = len(self.engine.counts)

        print('Dataset of {} instances'.format(size))
        if batch > size:
            self.FLAGS.batch = batch = size

        m = self.meta
        inp_H, inp_W, C = m['inp_size']
        m_net = m['net']
        active_H, active_W = m_net['active_height'], m_net['active_width']
        out_H, out_W, _ = m['out_size']
        scale = inp_H // out_H
        assert scale * out_H == inp_H and scale * out_W == inp_W

        train_ids = self.engine.training_ids()

        def get_training_image(tid):
            # import pdb; pdb.set_trace()
            # img = self.engine.training_image(tid)
            img = self.engine.training_mmap_image(tid)
            # mask = self.engine.training_mask(tid)
            mask = self.engine.training_mmap_mask(tid)
            # img *= mask[:, :, None]

            # rect = random_subimage_rect(mask, (active_H, active_W))
            h, w, _ = img.shape
            r = np.random.randint(h)
            c = np.random.randint(w)
            r0 = r - active_H // 2
            c0 = c - active_W // 2
            r1 = r0 + active_H
            c1 = c0 + active_W
            rect = Rect(r0, r1, c0, c1)

            density = self.engine.training_density(tid, scale=scale, rect=rect)
            rect = rect.reshape((inp_H, inp_W))
            img = subimage(img, rect)
            mask = subimage(mask[:, :, None], rect)
            img *= mask
            return img, density

        print('Finding unmasked areas...')
        mask_area = self.engine.training_mask_area(train_ids, cache='data/mask_area.npy')
        mask_area = np.array(mask_area)
        print('done')
        repeats = (mask_area / (active_H * active_W) + 0.5).astype(np.int)
        repeated_ids = np.repeat(train_ids, repeats)

        batch_per_epoch = len(repeated_ids) // batch

        for i in range(self.FLAGS.epoch):
            shuffled_ids = np.random.permutation(repeated_ids)
            for b in range(batch_per_epoch):
                batch_ids = shuffled_ids[b*batch:b*batch+batch]
                # foo = get_training_image(batch_ids)
                batch_pairs = map(get_training_image, batch_ids)
                images, densities = zip(*batch_pairs)  # list(tuple) -> tuple(list)
                images = np.stack(images)
                densities = np.stack(densities)
                feed = {
                    'density': densities
                }
                yield images, feed

            print('Finish {} epoch(es)'.format(i + 1))

    def preprocess(self, im, allobj=None):
        pass

    def loss(self, net_out):
        m = self.meta
        inp_H, inp_W, _ = m['inp_size']
        out_H, out_W, out_C = m['out_size']
        scale = inp_H // out_H
        m_net = m['net']
        active_H, active_W = m_net['active_height'], m_net['active_width']
        active_H //= scale
        active_W //= scale
        shape = [None, active_H, active_W, out_C]
        _density = tf.placeholder(tf.float32, shape, name="density")
        self.placeholders = {
            'density': _density
        }
        self.fetch += [_density]
        dh = out_H - active_H
        dw = out_W - active_W
        active_out = net_out[:, dh // 2:-dh // 2,
                             dw // 2:-dw // 2, :]
        d = active_out - _density
        self.loss = tf.sqrt(tf.reduce_mean(d * d))
        tf.summary.scalar('{} loss'.format(m['model']), self.loss)

    def is_inp(self, name):
        return os.path.splitext(name)[1] in ["png", "jpg"]

    def postprocess(self, net_out, im, save=True):
        textBuff = ','.join([str(n) for n in np.sum(net_out, axis=(0, 1))])
        img_name = os.path.basename(im)
        csv_path = os.path.splitext(img_name)[0] + ".csv"
        with open(csv_path, 'w') as f:
            f.write(textBuff)

    def resize_input(self, im):
        return im

    # def findboxes(self, net_out):
    #     pass

    # def process_box(self, b, h, w, threshold):
    #     pass


framework.types['[count]'] = CountFramework


class CountNet(TFNet):
    def return_predict(self, im):
        assert isinstance(im, np.ndarray), \
            'Image is not a np.ndarray'
        h, w, _ = im.shape
        im = self.framework.resize_input(im)
        this_inp = np.expand_dims(im, 0)
        feed_dict = {self.inp: this_inp}

        out = self.sess.run(self.out, feed_dict)[0]
        return np.sum(out, axis=(0, 1))

def main(args):
    # import pdb; pdb.set_trace()
    FLAGS = argHandler()
    FLAGS.setDefaults()
    FLAGS.parseArgs(args)

    # make sure all necessary dirs exist
    def _get_dir(dirs):
        for d in dirs:
            this = os.path.abspath(os.path.join(os.path.curdir, d))
            if not os.path.exists(this):
                os.makedirs(this)
    _get_dir([FLAGS.imgdir, FLAGS.binary, FLAGS.backup,
             os.path.join(FLAGS.imgdir, 'out'), FLAGS.summary])

    # fix FLAGS.load to appropriate type
    try:
        FLAGS.load = int(FLAGS.load)
    except:
        pass

    tfnet = CountNet(FLAGS)

    if FLAGS.train:
        print('Enter training ...')
        tfnet.train()
        if not FLAGS.savepb:
            exit('Training finished, exit.')

    if FLAGS.savepb:
        print('Rebuild a constant version ...')
        tfnet.savepb()
        exit('Done')

    tfnet.predict()

# example command line:
# > python density-nn.py --model count.cfg --load yolo.weights --train --gpu 0.9 --batch 24 --ntrain 13 --nload 40
# > python density-nn.py --model count.cfg --load 83 --train --batch 24 --ntrain 13 --gpu 0.9



if __name__ == '__main__':
    main(sys.argv)
