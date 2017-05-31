import os
import pandas as pd
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image

from point_density import point_density, downsample_sum


class Engine(object):
    def __init__(self,
                 data_dir='./data',
                 img_shape=(3744, 5632),
                 sigma=10):
        self.data_dir = data_dir
        self.points = pd.read_csv(self.data_path('coords-clean.csv'))
        self.counts = pd.read_csv(self.data_path('train-clean.csv'))
        self.class_names = ['adult_males',     # 0
                            'subadult_males',  # 1
                            'adult_females',   # 2
                            'juveniles',       # 3
                            'pups']            # 4
        self.classes = range(len(self.class_names))
        self.class_by_name = dict(zip(self.class_names,
                                      self.classes))
        self.img_shape = img_shape
        self.sigma = sigma

    def data_path(self, path):
        return '{}/{}'.format(self.data_dir, path)

    def pad_image(self, img):
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

    def training_image_path(self, tid, mode=''):
        for ext in ['jpg', 'png']:
            im_path = '{}/Train{}/{}.{}'.format(self.data_dir, mode, tid, ext)
            if os.path.isfile(im_path):
                return im_path
        return None

    def training_image(self, tid, mode=''):
        im_path = self.training_image_path(tid, mode)
        if im_path is not None:
            img = mpimg.imread(im_path)
            img = self.pad_image(img)
            return img
        return None

    def training_points(self, tid, cls=None):
        def transform_points(points):
            img_path = self.training_image_path(tid)
            with Image.open(img_path) as img:
                width, height = img.size
            if height > width:
                points[:, [0, 1]] = points[:, [1, 0]]
                height, width = width, height
            dh = self.img_shape[0] - height
            dw = self.img_shape[1] - width
            return points + np.array([[dh // 2, dw // 2]])
        if cls is not None:
            points = self.points[(self.points.tid == tid) &
                                 (self.points.cls == cls)]\
                         .as_matrix(['row', 'col'])
            return transform_points(points)
        else:
            points = self.points[self.points.tid == tid]\
                         .as_matrix(['cls', 'row', 'col'])
            points[:, 1:] = transform_points(points[:, 1:])
            return points

    def training_density(self, tid, cls=None, scale=32):
        if cls is None:
            return np.stack([self.training_density(tid, cls)
                             for cls in self.classes],
                            axis=-1)
        points = self.training_points(tid, cls)
        # h, w = self.img_shape
        # shape = h // scale, w // scale
        # points = points / scale
        # sigma = self.sigma / scale
        density = point_density(points, self.sigma, self.img_shape)
        return downsample_sum(density, scale)

    def display_locations(self, tid, cls=None, window_size=40):
        coords = self.training_points(tid, cls)
        if cls is None:
            coords = coords[:, 1:3]
        img = self.training_image(tid, 'Dotted')

        mask = np.zeros((self.img_shape[0], self.img_shape[1], 3),
                        dtype=np.bool)
        for ind in range(len(coords)):
            row_min = max(coords[ind][0] - window_size, 0)
            row_max = min(coords[ind][0] + window_size, self.img_shape[0])
            col_min = max(coords[ind][1] - window_size, 0)
            col_max = min(coords[ind][1] + window_size, self.img_shape[1])
            mask[row_min:row_max, col_min:col_max, :] = True

        plt.imshow(mask * img)
        plt.show()

    def extract_pictures(self, num_pictures=100, window_size=25):
        #this extracts num_pictures different pictures of sea lions in square images of radius window_size
        #and saves them to their respective folders
        assert num_pictures >= 1
        assert window_size >= 1
        for sealion_type in self.class_names:
            if (os.path.isdir('{}/{}'.format(self.data_dir, sealion_type)) == False):
                os.mkdir('{}/{}'.format(self.data_dir, sealion_type))

        class_counter = np.zeros(len(self.class_names), dtype='int')
        
        #iterate through all the files in the Training directory
        #and extract the pictures of the sea lions.
        for filename in os.listdir('{}/Train'.format(self.data_dir)):
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    tid = int(filename[:len(filename)-4])
                    coords = self.training_points(tid)
                    img = self.training_image(tid)
                    maxrow = self.img_shape[0] - window_size - 1
                    maxcol = self.img_shape[1] - window_size - 1
                    for c in coords:
                        if ((window_size <= c[1] <= maxrow) and (window_size <= c[2] <= maxcol)):
                            crop = img[c[1] - window_size: c[1]+window_size, c[2]-window_size: c[2]+window_size, :]
                            class_counter[c[0]] += 1
                            mpimg.imsave('{}/{}/{}.jpg'.format(self.data_dir, self.class_names[c[0]], str(class_counter[c[0]])), crop)
                        if (class_counter.sum() >= num_pictures):
                            break
                if (class_counter.sum() >= num_pictures):
                    break
                


                
        
        
