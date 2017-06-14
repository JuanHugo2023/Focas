########################################################################################
# Davi Frossard, 2016                                                                  #
# VGG16 implementation in TensorFlow                                                   #
# Details:                                                                             #
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
#                                                                                      #
# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md     #
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow      #
########################################################################################

import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
from imagenet_classes import class_names
import random
import os
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from scipy import ndimage
import io
import tkinter as tk
from math import ceil
from rect import Rect





class batch_handler():
    def __init__(self,
                 data_dir='./data',
                 padded_shape=None,
                 img_shape=(3744, 5632)):
        self.data_dir = data_dir
        self.points = pd.read_csv(self.data_path('coords-clean.csv'))
        self.counts = pd.read_csv(self.data_path('train-clean.csv'))

        #Check to see if there's a validation.csv
        #If yes, load it into self.validation_ids
        #If no, load 10% of the training set into self.validation_ids
        #       and create a new validation.csv file for later use

        #note: self.validation_ids contains only the ids of the validation set,
        #      not the information about the validation set itself

        try:
            self.validation_ids = pd.read_csv(self.data_path('validation.csv')).as_matrix()[:,0]
        except:
            ids = np.array(self.counts.index).copy()
            np.random.shuffle(ids)
            self.validation_ids = ids[:int(0.1 * len(ids))]
            self.validation_ids.sort()
            pd.Series(self.validation_ids).to_csv(self.data_path('validation.csv'), index=False)
            print('WARNING: no validation set found.')
        self.training_ids = np.array(list(set(self.counts.index) - set(self.validation_ids)))
        self.mismatched = [3, 7, 9, 21, 30, 34,71, 81, 89, 97, 151, 184, 215, 234, 242, 268, 290, 311, 331, 344, 380, 384, 406, 421, 469, 475, 490, 499, 507, 530, 531, 605, 607, 614, 621, 638, 644, 687, 712, 721, 767, 779, 781, 794, 800, 811, 839, 840, 869, 882, 901, 903, 905, 909, 913, 927, 946]
    
        self.training_ids = self.training_ids[[(tid not in self.mismatched) for tid in self.training_ids]]
        self.validation_ids = self.validation_ids[[(tid not in self.mismatched) for tid in self.validation_ids]]
        self.unsorted_training_ids = self.training_ids.copy()
        self.training_ids.sort()
        random.shuffle(self.unsorted_training_ids)

        #At this point, we have
        #   self.validation_ids
        #   self.training_ids


        self.class_names = ['adult_males',     # 0
                            'subadult_males',  # 1
                            'adult_females',   # 2
                            'juveniles',       # 3
                            'pups']            # 4
        self.classes = range(len(self.class_names))
        self.class_by_name = dict(zip(self.class_names,
                                      self.classes))
        self.padded_shape = padded_shape
        self.img_shape = img_shape


        #below is not in Benson's engine.py
        self.picture_counter = 0

    def data_path(self, path):
        return '{}/{}'.format(self.data_dir, path)

    def load_counts(self, path):
        return pd.read_csv(path, index_col=0)

    def pad_image(self, img):
        if self.padded_shape is None:
            return img
        dh = self.img_shape[0] - img.shape[0]
        dw = self.img_shape[1] - img.shape[1]
        assert dh >= 0
        assert dw >= 0
        if dh > 0 or dw > 0:
            def pad_width(d):
                return (d // 2, d - d // 2)
            img = np.pad(img,
                         (pad_width(dh), pad_width(dw), (0, 0)),
                         mode='constant',
                         constant_values=0)
        return img

    def create_npy_images(self, tids=None):
        #This takes the self.training_ids
        #   and saves them as .npy type files
        #   (does not mask them)
        if tids is None:
            tids = self.training_ids
        for tid in tids:
            img = self.training_image(tid)
            path = self.training_image_path(tid)
            path = os.path.splitext(path)[0] + '.npy'
            np.save(path, img)
            print("saved {}".format(path))

    def create_masked_npy_images(self, tids=None):
        #This takes the self.training_ids, masks them,
        #   and saves them as .npy type files
        #   (does not mask them)
        if tids is None:
            tids = self.training_ids
        for tid in tids:
            img = self.training_image(tid, masked=True)
            path = self.training_image_path(tid)
            path = os.path.splitext(path)[0] + '.npy'
            np.save(path, img)
            print("saved {}".format(path))
                
            

    def training_mmap_image(self, tid):
        #Loads the tid image as a memory-mapped image
        #   along the way, saves the image as a .npy file
        im_path = self.training_image_path(tid)
        npy_path = os.path.splitext(im_path)[0] + '.npy'
        if not os.path.isfile(npy_path):
            img = self.training_image(tid)
            np.save(npy_path, img)
        return np.load(npy_path, mmap_mode='r')

    def training_mmap_mask(self, tid):
        #Loads the mask as a memory-mapped image
        #   along the way, it saves the mask as a .mask.npy file
        im_path = self.training_image_path(tid)
        npy_path = os.path.splitext(im_path)[0] + '.mask.npy'
        if not os.path.isfile(npy_path):
            mask = self.training_mask(tid)
            np.save(npy_path, mask)
        return np.load(npy_path, mmap_mode='r')

    def training_masked_image(self, tid):
        #Returns the mask
        img = self.training_mmap_image(tid)
        mask = self.training_mmap_mask(tid)
        mask = mask[:,:,None]
        return img * mask

    def training_image_path(self, tid, dotted=False):
        mode = 'Dotted' if dotted else ''
        for ext in ['npy', 'jpg', 'png']:
            im_path = '{}/Train{}/{}.{}'.format(self.data_dir, mode, tid, ext)
            if os.path.isfile(im_path):
                return im_path
        return None

    def _training_image(self, tid, dotted=False):
        im_path = self.training_image_path(tid, dotted)
        if im_path is None:
            return None
        if im_path[-4:] == ".npy":
            img = np.load(im_path)
        else:
            img = mpimg.imread(im_path)
        return self.pad_image(img)

    def training_image(self, tid, dotted=False, masked=False):
        img = self._training_image(tid, dotted)
        if img is None:
            return None
        if masked:
            mask = self.training_mask(tid)
            img *= mask[:, :, None]
        return img

    def training_mask(self, tid):
        #Returns a boolean mask of all pixels that are non-black
        #   the dotted image 
        dotted = self._training_image(tid, dotted=True)
        if dotted is not None:
            dotted_lum = dotted.sum(axis=2)
            return dotted_lum > 10


    def _training_mask_area(self, tid):
        # return self.training_mask(tid).sum()
        return self.training_mmap_mask(tid).sum()


    def training_mask_area(self, tid, cache=None):
        try:  # try to treat tid as an iterable
            iter(tid)
            if cache is not None:
                try:
                    return np.load(cache)
                except:
                    pass
            # with Pool() as pool:
            #     result = np.array(pool.map(self._training_mask_area, tid))
            result = np.array(list(map(self._training_mask_area, tid)))
            if cache is not None:
                np.save(cache, result)
            return result
        except TypeError:  # tid is not iterable
            return self._training_mask_area(tid)

    def training_image_shape(self, tid):
        if self.padded_shape is not None:
            return self.padded_shape
        img_path = self.training_image_path(tid)
        if os.path.splitext(img_path)[1] == '.npy':
            return self.training_mmap_image(tid).shape[:2]
        else:
            with Image.open(img_path) as img:
                width, height = img.size
                return height, width



    def training_points(self, tid, cls=None, rect=None, expand=0):
        h, w = self.training_image_shape(tid)
        rect = Rect(0, h, 0, w)


        points_in_rect = (self.points.tid == tid) & \
                         (rect.row_min <= self.points.row + expand) & \
                         (self.points.row < rect.row_max + expand) & \
                         (rect.col_min <= self.points.col + expand) & \
                         (self.points.col < rect.col_max + expand)

        if cls is not None:
            points = self.points[(self.points.cls == cls) &
                                 points_in_rect] \
                         .as_matrix(['row', 'col'])
            return rect.transform(points)
        else:
            points = self.points[points_in_rect] \
                         .as_matrix(['cls', 'row', 'col'])
            points[:, 1:] = rect.transform(points[:, 1:])
            return points
        
    def test_image_path(self, test_id):
        im_path = '{}/Test/{}.jpg'.format(self.data_dir, test_id)
        if os.path.isfile(im_path):
            return im_path
        return None

    def test_image(self, tid):
        im_path = self.test_image_path(tid)
        if im_path is None:
            return None
        return mpimg.imread(im_path)

    def test_ids(self):
        files = os.listdir(os.path.join(self.data_dir, 'Test'))
        files = (os.path.splitext(f) for f in files)
        ids = [int(id) for (id, ext) in files if ext == '.jpg']
        ids.sort()
        return ids

    def test_counts(self, path='test.csv'):
        try:
            return pd.read_csv(path, index_col=0)
        except:
            df = pd.DataFrame([], columns=self.class_names)
            df.index.name = 'test_id'
            return df


    def extract_blanks(self, num_pictures, window_diam, progress_indicator=True):
        #this should return a matrix of an appropriate size full of vacant pictures
        pics = np.zeros((num_pictures, window_diam, window_diam, 3), dtype='uint8')
        counter = 0
        
        #use: training_image(self, tid, dotted=False, masked=False):
        #use: training_points(self, tid, cls=None, rect=None, expand=0):

        shuffled_tids = self.training_ids.copy()
        np.random.shuffle(shuffled_tids)

        for tid in shuffled_tids:
            if (progress_indicator == True):
                print('Processing {}'.format(tid))
            img = self.training_image(tid)
            coords = self.training_points(tid)

            maxrow = (img.shape[0] // window_diam) - 1
            maxcol = (img.shape[1] // window_diam) - 1

            mask = np.ones((maxrow, maxcol), dtype='bool')
            
            for c in (coords[:, 1:] // window_diam):
                if (c[0] < maxrow and c[1] < maxrow):
                    mask[c[0], c[1]] = False

            for b in np.argwhere(mask):
                if ((b[0] % 4) or (b[1] % 4)):
                    mask[b[0], b[1]] = False

            for b in np.argwhere(mask):
                r1, c1 = b[0] * window_diam, b[1] * window_diam
                pics[counter] = img[r1:r1+window_diam, c1:c1+window_diam, :]
                counter += 1
                if (counter >= num_pictures):
                     break
            if (counter >= num_pictures):
                break
        return pics

    def extract_mix(self, num_vacant, num_pictures, max_per_window, window_diam=224, progress_indicator=True):
        #this will take num_vacant pictures from 'vacant' class and num_pictures from each of the 5 classes
        num_total = num_vacant + (5 * num_pictures)
        label_length = (1 + (max_per_window * 5))
        pics = np.zeros((num_total, window_diam, window_diam, 3), dtype='uint8')
        #labels = np.zeros((num_total, 6), dtype='float32')
        labels = np.zeros((num_total, label_length), dtype='float32')
        print('Getting the vacant pictures')
        pics[:num_vacant] = self.extract_blanks(num_vacant, window_diam, progress_indicator=False)
        labels[:num_vacant, 0] = 1

        intra_picture_class_counter = np.zeros(5, dtype=int)
        
        window_rad = window_diam // 2
        total_counter = num_vacant

        shuffled_tids = self.training_ids.copy()
        np.random.shuffle(shuffled_tids)

        for class_num in range(5):
            class_counter = 0
            print('Processing class {}'.format(class_num))
            for tid in shuffled_tids:
                if (progress_indicator == True):
                    print('Processing {} of class {}, finished {}/{}'.format(tid, class_num, class_counter, num_pictures))
                img = self.training_image(tid)
                coords = self.training_points(tid)
                maxrow = img.shape[0] - window_rad - 1
                maxcol = img.shape[1] - window_rad - 1

                for c in coords:
                    if (c[0] == class_num and (window_rad <= c[1] <= maxrow) and (window_rad <= c[2] <= maxcol)):
                        pics[total_counter] = img[c[1] - window_rad: c[1]-window_rad + window_diam, c[2]-window_rad: c[2]-window_rad + window_diam, :]
                        intra_picture_class_counter = np.zeros(5, dtype=int)
                        for i in np.argwhere((c[1] - window_rad <= coords[:,1]) &
                                             (coords[:,1] <= c[1]+window_rad) &
                                             (c[2] - window_rad <= coords[:,2]) &
                                             (coords[:,2] <= c[2]+window_rad)):
                            intra_picture_class_counter[coords[i[0], 0]] += 1 
                                
                            #labels[total_counter, coords[i[0], 0] + 1] += 1 #could change this to a +1 if wanted to count
                            
                        for i in range(5):
                            num_to_change = min(max_per_window, intra_picture_class_counter[i])
                            #print('dtypes are {}, {}'.format(type(i), type(num_to_change)))
                            labels[total_counter, (max_per_window * i) + 1: (max_per_window * i) + 1 + num_to_change] = 1
                            
                        #labels[total_counter] = labels[total_counter] / labels[total_counter].sum() #changes it into a probability distribution
                        total_counter += 1
                        class_counter += 1
                    if (class_counter >= num_pictures):
                        break
                if (class_counter >= num_pictures):
                    break
        return pics, labels                       

                
                



    def initialize_mixed_training_set(self, num_vacant, num_per_class, max_per_window, window_diam=224):
        sea_lion_classes = {'adult_males':0,     # 0
                            'subadult_males':1,  # 1
                            'adult_females':2,   # 2
                            'juveniles':3,       # 3
                            'pups':4}
        self.training_data, self.training_labels = self.extract_mix(num_vacant, num_per_class, max_per_window, progress_indicator=True)

        self.randomized_order = list(range(len(self.training_data)))
        random.shuffle(self.randomized_order)

        #self.tf_training_labels = tf.to_int64(self.training_labels)
        self.current_counter = 0        
        
        
        

    def serve_batch(self, batch_size):
        #this should take in a batch size, and get that many labelled pictures to train on
        self.current_counter += batch_size
        if (self.current_counter >= len(self.randomized_order)):
            self.current_counter = batch_size
            print('going back to the beginning of the data')
        return self.training_data[self.randomized_order[self.current_counter - batch_size: self.current_counter]], self.training_labels[self.randomized_order[self.current_counter - batch_size: self.current_counter]]


    def serve_image(self):
        tid = self.unsorted_training_ids[self.picture_counter]
        img = self.training_image(tid)
        coords = self.training_points(tid)        
        self.picture_counter += 1
        if (self.picture_counter == len(self.training_ids)):
            self.picture_counter = 0
            print('Out of pictures to serve, starting from the beginning')
        return img, coords
        

    def serve_validation_image(self, i):
        tid = self.validation_ids[i]
        img = self.training_image(tid)
        coords = self.training_points(tid)        
        return img, coords
    

class vgg16:
    def __init__(self, imgs, labels=None, weights=None, sess=None):
        self.imgs = imgs
        self.labels = labels
        self.convlayers()
        self.fc_layers()
        self.probs = tf.nn.softmax(self.fc3l)
        if weights is not None and sess is not None:
            self.load_weights(weights, sess)


    def convlayers(self):
        self.parameters = []

        # zero-mean input
        # this takes the image and subtracts a number from each of the RGB channels, effectively making the mean equal to 0
        # it stores the result in the local variable 'images'
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = self.imgs-mean

        # conv1_1
        # sets up a first convolutional layer with 64 channels, using a 3x3(x3) kernel
        # conv -> conv + bias -> relu -> saved into self.conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

    def fc_layers(self):
        # fc1
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            fc1w = tf.Variable(tf.truncated_normal([shape, 4096],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                 trainable=True, name='biases')
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
            self.fc1 = tf.nn.relu(fc1l)
            self.parameters += [fc1w, fc1b]

        # fc2
        with tf.name_scope('fc2') as scope:
            fc2w = tf.Variable(tf.truncated_normal([4096, 4096],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                 trainable=True, name='biases')
            fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
            self.fc2 = tf.nn.relu(fc2l)
            self.parameters += [fc2w, fc2b]

        # fc3
        with tf.name_scope('fc3') as scope:
            fc3w = tf.Variable(tf.truncated_normal([4096, 1000],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc3b = tf.Variable(tf.constant(1.0, shape=[1000], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)
            self.parameters += [fc3w, fc3b]

    def load_weights(self, weight_file, sess):
        print('Loading weights from {}'.format(weight_file))
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            print (i, k, np.shape(weights[k]))
            sess.run(self.parameters[i].assign(weights[k]))

    def reset_final_layer(self, max_per_window):
        num_classes = 1 + (max_per_window * 5)

        with tf.name_scope('fc3_new') as scope:
            fc3w_new = tf.Variable(tf.truncated_normal([4096, num_classes],
                                                           dtype=tf.float32,
                                                           stddev=1e-1), name='weights')
            fc3b_new = tf.Variable(tf.constant(1.0, shape=[num_classes], dtype=tf.float32), trainable=True, name='biases')
            self.fc31_new = tf.nn.bias_add(tf.matmul(self.fc2, fc3w_new), fc3b_new)
            self.parameters = self.parameters[:-2] + [fc3w_new, fc3b_new]
            #self.probs = tf.nn.softmax(self.fc31_new)
            self.sigmoid_output = tf.nn.sigmoid(self.fc31_new)
            sess.run(tf.variables_initializer([self.parameters[-2], self.parameters[-1]]))
        

class visualizer():

    def __init__(self, sess, vgg, bh, max_per_window = 30):
        self.height = 600
        self.width = 800
        self.root = tk.Tk()
        self.f = tk.Frame(self.root, height=self.height, width=self.width)
        self.f.pack()
        self.canv = tk.Canvas(self.f, height=self.height, width=self.width)
        self.canv.pack()
        self.counter = 0
        self.bh = bh
        self.vgg = vgg
        self.sess = sess
        self.sliding_window_total_count = np.zeros(5)
        self.max_per_window = max_per_window
        
    def show_image(self, x_pos, y_pos, img):
        im_img = Image.fromarray(img)
        self.phot_img = ImageTk.PhotoImage(im_img)
        self.canv.create_image(x_pos, y_pos, image=self.phot_img, anchor=tk.NW)
        
    def show_prob(self, img):
        #prob = self.sess.run(self.vgg.probs, feed_dict={self.vgg.imgs:img})[0]
        prob = self.sess.run(self.vgg.sigmoid_output, feed_dict={self.vgg.imgs:img})[0]
        self.canv.create_text(10, 300, text='Prediction: {}'.format(self.one_hot_to_numbers(prob)), anchor=tk.NW)

    def update(self):
        self.canv.delete('all')
        img, labels = self.bh.serve_batch(1)
        self.show_image(10, 10, img[0])
        self.show_prob(img)
        self.canv.create_text(10, 320, text='     Truth: {}'.format(self.one_hot_to_numbers(labels[0])), anchor=tk.NW)

    def one_hot_to_numbers(self, y):
        opt = np.zeros(5)
        for i in range(5):
            opt[i] = y[(self.max_per_window * i) + 1: (self.max_per_window * (i+1)) + 1].sum()
        return opt

    def done_sliding_window(self, coords):
        print('outside of u_s_d')
        self.canv.delete('window')
        self.canv.delete('sliding_predictions')

        picture_tally_truth = np.zeros(5)
        U, V = np.unique(coords[:, 0], return_counts=True)
        for i in range(len(U)):
            picture_tally_truth[U[i]] = V[i]
        self.canv.create_text(10, 410, text='Prediction: {}'.format(self.sliding_window_total_count), anchor=tk.NW)
        self.canv.create_text(10, 460, text='     Truth: {}'.format(picture_tally_truth), anchor=tk.NW)
        

    def update_sliding_window(self, img, x, y, stride, coords, window_diam=224):
        #amount_to_wait is the number of milliseconds that the screen pauses if it finds a square with sea lions in it
        amount_to_wait = 1000
        
        x_scale_factor = img.shape[1] / 600
        y_scale_factor = img.shape[0] / 400
        window = img[y:y+window_diam, x:x+window_diam]
        sig_output = self.sess.run(self.vgg.sigmoid_output, feed_dict={self.vgg.imgs:[window]})[0]
        square_tally = self.one_hot_to_numbers(sig_output)

        nwc_x = 10 + ((600 * x) // img.shape[1])
        nwc_y = 10 + ((400 * y) // img.shape[0])

        scaled_w_x = ((600 * window_diam) // img.shape[1])
        scaled_w_y = ((400 * window_diam) // img.shape[0])
        scaled_wd = window_diam // 10
    
        
        self.canv.delete('window')
        self.canv.delete('sliding_predictions')

        self.canv.create_line(nwc_x, nwc_y, nwc_x+scaled_w_x, nwc_y, fill='yellow', tag='window')
        self.canv.create_line(nwc_x+scaled_w_x, nwc_y, nwc_x+scaled_w_x, nwc_y+scaled_w_y, fill='yellow', tag='window')
        self.canv.create_line(nwc_x+scaled_w_x, nwc_y+scaled_w_y, nwc_x, nwc_y+scaled_w_y, fill='yellow', tag='window')
        self.canv.create_line(nwc_x, nwc_y+scaled_w_y, nwc_x, nwc_y, fill='yellow', tag='window')

        self.canv.create_text(10, 410, text='Window Prediction: {}'.format(square_tally), anchor=tk.NW, tag='sliding_predictions')
        square_tally[square_tally < .4] = 0
        self.canv.create_text(10, 430, text='Cutoff Window Prediction: {}'.format(square_tally), anchor=tk.NW, tag='sliding_predictions')
        self.sliding_window_total_count += square_tally
        self.canv.create_text(10, 460, text='Running Tally for Picture: {}'.format(self.sliding_window_total_count), anchor=tk.NW, tag='sliding_predictions')

        
        if (square_tally.sum() == 0):
            amount_to_wait = 10

        x = x + stride
        if (x + window_diam >= img.shape[1]):
            y = y + stride
            x = 0
        if (y + window_diam < img.shape[0]):
            self.canv.after(amount_to_wait, lambda: self.update_sliding_window(img, x, y, stride, coords, window_diam))
        else:
            self.canv.after(amount_to_wait, lambda: self.done_sliding_window(coords))
                    

    def sliding_window(self, tid):
        self.canv.delete('all')
        window_diam = 224
        fraction_of_window = 1 
        stride = window_diam // fraction_of_window
        img = bh.training_image(tid)
        coords = bh.training_points(tid)              
        self.sliding_window_total_count = np.zeros(5)

        im_img = Image.fromarray(img)
        resized_img = im_img.resize((600, 400))
        self.phot_img = ImageTk.PhotoImage(resized_img)


        
        self.canv.create_image(10, 10, image=self.phot_img, anchor=tk.NW)

        self.canv.after(100, lambda: self.update_sliding_window(img, 0, 0, stride, coords, window_diam))


        

      
        
        
        
            

def train_cnn(sess, vgg, bh, num_steps, learning_rate=0.002):
    batch_size = 2
    #learning_rate=.0001
    #learning_rate=.005
    #learning_rate=2


    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=vgg.labels, logits=vgg.fc31_new, name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    global_step = tf.Variable(0, name='global_step', trainable=False)
    sess.run(tf.variables_initializer([global_step]))
    train_op = optimizer.minimize(loss, global_step=global_step)

    for step in range(num_steps):
        X, Y = bh.serve_batch(batch_size)
        _, loss_value, sig_output = sess.run([train_op, loss, vgg.sigmoid_output], feed_dict={vgg.imgs: X, vgg.labels: Y})
        print('On step {0}, loss value = {1:.5f}'.format(step, loss_value))

        


def test_cnn(sess, vgg, bh, pictures=None, window_diam=224, max_per_window=30, stride=224, pup_cap=True, adult_male_cap=True):
    if pictures is None:
        pictures = bh.validation_ids

    window_tally = np.zeros(5)
    all_predictions = np.zeros((len(pictures), 5))
    all_truths = np.zeros((len(pictures), 5))
    counter = 0
    
    for tid in pictures:
        print('\nAnalyzing picture {} of {}, tid={}'.format(counter+1, len(pictures), tid))
        img = bh.training_image(tid)
        coords = bh.training_points(tid)

        #calculate true counts
        U, V = np.unique(coords[:, 0], return_counts=True)
        for i in range(len(U)):
            all_truths[counter, U[i]] = V[i]

        #predict counts with a sliding window
        x = 0
        y = 0
        while(y + window_diam < img.shape[0]):
            window = img[y:y+window_diam, x:x+window_diam]

            #count the sea lions in the window, threshold it at 0.45, then add the result to all_predictions
            sig_output = sess.run(vgg.sigmoid_output, feed_dict={vgg.imgs:[window]})[0]
            for j in range(5):
                window_tally[j] = sig_output[(max_per_window * j) + 1: (max_per_window * (j+1)) + 1].sum()
            window_tally[window_tally < .45] = 0
            all_predictions[counter, :] += window_tally

            #move the window
            x = x + stride
            if (x + window_diam >= img.shape[1]):
                y = y + stride
                x = 0

        #Apply corrections to prevent counting seals as pups, and avoid counting rocks as adult males   
        if ((pup_cap is True) and (all_predictions[counter, 4] > all_predictions[counter, 2])):
            all_predictions[counter, 4] = all_predictions[counter, 2]
        if ((adult_male_cap is True) and (all_predictions[counter, 0] > 50)):
            all_predictions[counter, 0] = 50

        #Print the summary comparing predictions to truth
        error = all_predictions[counter, :] - all_truths[counter, :]
        print('Prediction: {}'.format(all_predictions[counter, :]))
        print('Truth: {}'.format(all_truths[counter, :]))
        print('Squared Error: {}'.format(error * error))
        counter += 1

    all_errors = all_predictions - all_truths
    overall_score = np.sqrt(((all_errors * all_errors).sum(axis=0)) / (len(pictures)))
    print('Overall score: mean of {0} is {1:.4f}'.format(overall_score, (overall_score.sum())/5))

    return all_predictions, all_truths

            
            

            
        



    


if __name__ == '__main__':
    np.set_printoptions(precision=2, suppress=True)

    #The following option changes the architecture of the NN, it determines how many output bits to allocate for each class
    #   if you change this to a number other than 30, you cannot use the saved checkpoint
    max_per_window = 30
    
    #These options are for training:
    #   training_mode is True if you want the NN to train, false otherwise
    #   checkpoint_mode is True if you want to save your progress, false otherwise
    #   num_vacant is the number of pictures to put in the training set with 0 sea lions
    #   num_per_class is the number of pictures centered on sea lions (of each class) to put in the training set
    #       This means there will be num_vacant + 5(num_per_class) pictures total
    #   num_steps is the total number of batches to process during training
    training_mode = True
    restore_from_checkpoint_mode = True
    save_to_checkpoint_mode = True
    num_vacant = 40
    num_per_class = 40
    num_steps = 20
    learning_rate = 0.002
    

    #Initialize the tensorflow session, create the computational graph, and change the final layer
    #If restore_from_checkpoint_mode is True, it will load the weights from the given checkpoint file
    #   otherwise, the weights for the last layer will be randomized
    #The placeholders imgs and labs hold the inputs (subimages and labels, respectively) to the NN
    #   Our labels will have one position for 'vacant', and max_per_window positions for each of the 5 classes
    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    labs = tf.placeholder(tf.float32, [None, (1 + (max_per_window * 5))])
    vgg = vgg16(imgs, labs, 'vgg16_weights.npz', sess)
    vgg.reset_final_layer(max_per_window)
    saver = tf.train.Saver()
    if (restore_from_checkpoint_mode is True):
        saver.restore(sess, 'final_weights_p01_max30.ckpt')

    #bh supplies the pictures.
    bh = batch_handler()
    if (training_mode is True):
        bh.initialize_mixed_training_set(num_vacant, num_per_class, max_per_window, window_diam=224)    
        train_cnn(sess, vgg, bh, num_steps, learning_rate=learning_rate)
    
    #save a checkpoint
    if (save_to_checkpoint_mode is True):
        print('Saving')
        saver.save(sess, os.path.join(os.getcwd(), 'final_weights_p01_max30.ckpt'))
        print('Done')

    #To run a sliding window visualization on tid 122:    
    #   v = visualizer(sess, vgg, bh)
    #   v.sliding_window(122)

    #If you've loaded training set images because training_mode is True, then you can
    #   look at a count on a random window with the command
    #   v = visualizer(sess, vgg, bh)
    #   v.update()

    


