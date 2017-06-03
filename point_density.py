import numpy as np


def downsample_sum(a, scale):
    new_shape = (a.shape[0] // scale, scale, a.shape[1] // scale, scale) + a.shape[2:]
    a = a.reshape(new_shape)
    return a.sum(axis=(1, 3))


def gauss1d(sigma, truncate=4.0):
    lw = int(sigma * truncate) + 1
    sd = sigma * sigma
    x = np.linspace(-lw, lw, 2*lw + 1)
    g = np.exp(-0.5 * x * x / sd)
    g /= g.sum()
    return g


def gauss2d(sigma, truncate=4.0):
    g = gauss1d(sigma, truncate)
    return np.outer(g, g)


def point_density(points, shape, filter):
    h, w = shape
    a = np.zeros(shape)
    fh, fw = filter.shape
    for y, x in points:
        y0 = np.clip(y - fh // 2, 0, h-1)
        y1 = np.clip(y + fh // 2, 0, h-1)
        x0 = np.clip(x - fw // 2, 0, w-1)
        x1 = np.clip(x + fw // 2, 0, w-1)
        fy0 = y0 - y + fh // 2
        fy1 = y1 - y + fh // 2
        fx0 = x0 - x + fw // 2
        fx1 = x1 - x + fw // 2
        a[y0:y1, x0:x1] += filter[fy0:fy1, fx0:fx1]
    return a
