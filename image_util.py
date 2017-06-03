import numpy as np

from rect import Rect


def subimage(img, rect):
    r0, r1, c0, c1 = rect
    r0_, r1_, c0_, c1_ = 0, r1-r0, 0, c1-c0
    h, w, n = img.shape
    subimage = np.zeros((r1_, c1_, n), dtype=img.dtype)

    def constrain(r, r_, h):
        if r < 0:
            r_ -= r
            r = 0
        if r >= h:
            r_ -= r-h+1
            r = h-1
        return r, r_

    r0, r0_ = constrain(r0, r0_, h)
    r1, r1_ = constrain(r1, r1_, h)
    c0, c0_ = constrain(c0, c0_, w)
    c1, c1_ = constrain(c1, c1_, w)
    subimage[r0_:r1_, c0_:c1_, :] += img[r0:r1, c0:c1, :]
    return subimage


def random_subimage_rect(mask, shape):
    """
    Get (row_min, row_max, col_min, col_max) for a
    random subimage of the training image, with probability proportional
    to the number of unmasked pixels
    """
    h, w = shape
    mask = np.pad(mask,
                  ((0, h), (0, w)),
                  mode='constant',
                  constant_values=0)
    mask_cum = np.cumsum(mask.astype(np.float32), axis=0)
    mask_cum = np.cumsum(mask_cum, axis=1)
    p = mask_cum
    p[h:, :] -= mask_cum[:-h, :]
    p[:, w:] -= mask_cum[:, :-w]
    p[h:, w:] += mask_cum[:-h, :-w]
    p /= h*w
    p_cum = np.cumsum(p)
    norm = p_cum[-1]
    x = np.random.rand() * norm
    i = np.searchsorted(p_cum, x)
    r_max, c_max = np.unravel_index(i, p.shape)
    r_min = r_max - h
    c_min = c_max - w
    r_max += 1
    r_min += 1
    return Rect(r_min, r_max, c_min, c_max)


def random_subimage(img, mask, shape):
    rect = random_subimage_rect(mask, shape)
    return subimage(img, rect), (rect[0], rect[2])
