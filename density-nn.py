
import numpy as np
import tensorflow as tf
import os
import sys

from darkflow.net import framework
from darkflow.net.build import TFNet
from darkflow.defaults import argHandler
from engine import Engine


class CountFramework(framework.framework):
    def constructor(self, meta, FLAGS):
        self.engine = Engine()
        self.meta = meta
        self.FLAGS = FLAGS
        self.fetch = []

    def parse(self, exclusive=False):
        pass

    def shuffle(self):
        batch = self.FLAGS.batch
        size = len(self.engine.counts)

        print('Dataset of {} instances'.format(size))
        if batch > size:
            self.FLAGS.batch = batch = size
        # import pdb; pdb.set_trace()

        batch_per_epoch = size // batch

        train_ids = self.engine.training_ids()

        for i in range(self.FLAGS.epoch):
            shuffled_ids = np.random.permutation(train_ids)
            for b in range(batch_per_epoch):
                batch_ids = shuffled_ids[b*batch:b*batch+batch]
                images = np.array([self.engine.training_masked_image(tid)
                                   for tid in batch_ids])
                densities = np.array([self.engine.training_density(tid, scale=32)
                                      for tid in batch_ids])
                feed = {
                    'density': densities
                }
                yield images, feed

            print('Finish {} epoch(es)'.format(i + 1))

    def preprocess(self, im, allobj=None):
        pass

    def loss(self, net_out):
        m = self.meta
        H, W, C = m['out_size']
        shape = [None, H, W, C]
        _density = tf.placeholder(tf.float32, shape, name="density")
        self.placeholders = {
            'density': _density
        }
        d = net_out - _density
        self.fetch += [_density]
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
    FLAGS = argHandler()
    FLAGS.setDefaults()
    FLAGS.parseArgs(args)

    # make sure all necessary dirs exist
    def _get_dir(dirs):
        for d in dirs:
            this = os.path.abspath(os.path.join(os.path.curdir, d))
            if not os.path.exists(this): os.makedirs(this)
    _get_dir([FLAGS.imgdir, FLAGS.binary, FLAGS.backup, 
             os.path.join(FLAGS.imgdir,'out'), FLAGS.summary])

    # fix FLAGS.load to appropriate type
    try: FLAGS.load = int(FLAGS.load)
    except: pass

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


if __name__ == '__main__':
    main(sys.argv)
