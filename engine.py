import os
import pandas as pd
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
from multiprocessing import Pool

from math import ceil

from point_density import point_density, downsample_sum, gauss2d
from rect import Rect

_engine = None


def get_engine():
    """Get a shared, global instance of engine.
    """
    return _engine


def init_engine(data_dir='./data',
                padded_shape=None,
                sigma=(40, 30, 30, 30, 10)):
    """Initialize a shared, global instance of engine.
    """
    global _engine
    _engine = Engine(data_dir, padded_shape, sigma)

class Engine(object):
    def __init__(self,
                 data_dir='./data',
                 padded_shape=None,
                 sigma=(40, 30, 30, 30, 10)):
        self.data_dir = data_dir
        self.points = pd.read_csv(self.data_path('coords-clean.csv'))
        self.counts = self.load_counts(self.data_path('train-clean.csv'))
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
        self.training_ids.sort()

        self.class_names = ['adult_males',     # 0
                            'subadult_males',  # 1
                            'adult_females',   # 2
                            'juveniles',       # 3
                            'pups']            # 4
        self.classes = range(len(self.class_names))
        self.class_by_name = dict(zip(self.class_names,
                                      self.classes))
        self.padded_shape = padded_shape
        try:
            self.sigma = tuple(sigma)
        except:
            self.sigma = (sigma,) * 5
        self.point_filter = tuple(gauss2d(s) for s in self.sigma)
        self._training_mmap_images = None

    def data_path(self, path):
        return '{}/{}'.format(self.data_dir, path)

    def load_counts(self, path):
        return pd.read_csv(path, index_col=0)

    def pad_image(self, img):
        if self.padded_shape is None:
            return img
        dh = self.padded_shape[0] - img.shape[0]
        dw = self.padded_shape[1] - img.shape[1]
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
        if tids is None:
            tids = self.training_ids
        for tid in tids:
            img = self.training_image(tid)
            path = self.training_image_path(tid)
            path = os.path.splitext(path)[0] + '.npy'
            np.save(path, img)
            print("saved {}".format(path))

    def training_mmap_image(self, tid):
        im_path = self.training_image_path(tid)
        npy_path = os.path.splitext(im_path)[0] + '.npy'
        if not os.path.isfile(npy_path):
            img = self.training_image(tid)
            np.save(npy_path, img)
        return np.load(npy_path, mmap_mode='r')

    def training_mmap_mask(self, tid):
        im_path = self.training_image_path(tid)
        npy_path = os.path.splitext(im_path)[0] + '.mask.npy'
        if not os.path.isfile(npy_path):
            mask = self.training_mask(tid)
            np.save(npy_path, mask)
        return np.load(npy_path, mmap_mode='r')

    # def training_mmap_density(self, tid, subsample=1):
    #     scale = 4
    #     im_path = self.training_image_path(tid)
    #     npy_path = os.path.splitext(im_path)[0] + '.density.npy'
    #     if not os.path.isfile(npy_path):
    #         density = self.training_density(tid, scale=scale)
    #         np.save(npy_path, density)
    #     density = np.load(npy_path, mmap_mode='r')
    #     h, w, c = density.shape
    #     subsample *= scale
    #     density = np.stack([density]*subsample, axis=2)
    #     density = np.stack([density]*subsample, axis=1)
    #     density = density.reshape((h*subsample, w*subsample, c))
    #     return density

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

    def training_padded_rect(self, tid):
        h, w = self.training_image_shape(tid)
        if self.padded_shape is None:
            return Rect(0, h, 0, w)
        padded_h, padded_w = self.padded_shape
        transposed = False
        if h > w:
            transposed = True
            h, w = w, h
        dh = padded_h - h
        dw = padded_w - w
        row_min = dh // 2
        row_max = row_min + h
        col_min = dw // 2
        col_max = col_min + w
        return Rect(row_min, row_max, col_min, col_max, transposed)

    def training_points(self, tid, cls=None, rect=None, expand=0):
        if rect is None:
            rect = self.training_padded_rect(tid)

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

    def training_density(self, tid, cls=None, rect=None, scale=1, subsample=1):
        if rect is None:
            rect = self.training_padded_rect(tid)
        if cls is None:
            return np.stack([self.training_density(tid, cls, rect=rect,
                                                   scale=scale,
                                                   subsample=subsample)
                             for cls in self.classes],
                            axis=-1)
        points = self.training_points(tid, cls,
                                      rect=rect,
                                      expand=self.sigma[cls] * 4)
        height, width = rect.shape()
        if subsample != 1:
            points *= subsample
            height *= subsample
            width *= subsample
        density = point_density(points, (height, width), self.point_filter[cls])
        if scale != 1:
            dh = ceil(density.shape[0] / scale) * scale - density.shape[0]
            dw = ceil(density.shape[1] / scale) * scale - density.shape[1]
            density = np.pad(density, [(dh // 2, dh - dh // 2),
                                       (dw // 2, dw - dw // 2)],
                             'constant', constant_values=0)
            density = downsample_sum(density, scale)
        return density

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

    def display_locations(self, tid, cls=None, window_size=40):
        coords = self.training_points(tid, cls)
        if cls is None:
            coords = coords[:, 1:3]
        img = self.training_image(tid)

        mask = np.zeros((img.shape[0], img.shape[1], 3),
                        dtype=np.bool)
        for ind in range(len(coords)):
            row_min = max(coords[ind][0] - window_size, 0)
            row_max = min(coords[ind][0] + window_size, img.shape[0])
            col_min = max(coords[ind][1] - window_size, 0)
            col_max = min(coords[ind][1] + window_size, img.shape[1])
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
                


                
        
        
