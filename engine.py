import os
import pandas as pd
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from point_density import point_density


class Engine(object):
    def __init__(self,
                 data_dir='./data',
                 #img_shape=(3744, 5632),
                 img_shape=(3328, 4992),
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
                         (pad_width(dh), pad_width(dw), (0,0)),
                         mode='constant',
                         constant_values=0)
        return img

    def training_image(self, tid, mode=''):
        for ext in ['jpg', 'png']:
            im_path = '{}/Train{}/{}.{}'.format(self.data_dir, mode, tid, ext)
            if os.path.isfile(im_path):
                img = mpimg.imread(im_path)
                img = self.pad_image(img)
                return img
        return None

    def training_points(self, tid, cls=None):
        if cls is not None:
            return self.points[(self.points.tid == tid) &
                               (self.points.cls == cls)]\
                       .as_matrix(['row', 'col'])
        else:
            return self.points[self.points.tid == tid]\
                       .as_matrix(['cls', 'row', 'col'])

    def training_density(self, tid, cls=None, scale=32):
        if cls is None:
            return np.stack([self.training_density(tid, cls)
                                  for cls in self.classes],
                                 axis=-1)
        points = self.training_points(tid, cls)
        h, w = self.img_shape
        shape = h // scale, w // scale
        points = points / scale
        sigma = self.sigma / scale
        density = point_density(points, sigma, shape)
        return density

    def display_locations(self, tid, cls=None, window_size=40):
        coords = self.training_points(tid, cls)
        if cls == None:
            coords = coords[:, 1:3]
        img = self.training_image(tid)

        mask = np.zeros((self.img_shape[0], self.img_shape[1], 3), dtype='bool')
        for ind in range(len(coords)):
            row_min = max(coords[ind][0] - window_size, 0)
            row_max = min(coords[ind][0] + window_size, self.img_shape[0])
            col_min = max(coords[ind][1] - window_size, 0)
            col_max = min(coords[ind][1] + window_size, self.img_shape[1])
            mask[row_min:row_max, col_min:col_max, :] = True

        plt.imshow(mask * img)
        plt.show()
        
            
        
