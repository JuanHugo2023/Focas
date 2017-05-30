import numpy as np


def downsample_sum(a, scale):
    new_shape = (a.shape[0] // scale, scale, a.shape[1] // scale, scale) + a.shape[2:]
    a = a.reshape(new_shape)
    return a.sum(axis=(1, 3))


def gauss1d(mu, sigma, truncate, i_min, i_max):
    sigma = float(sigma)
    lw = int(sigma * truncate) + 1
    sd = sigma * sigma
    i = int(mu + 0.5)
    i0 = max(i_min, i-lw)
    i1 = min(i_max-1, i+lw)
    x = np.linspace(i0-mu, i1-mu, i1-i0+1)
    g = np.exp(-0.5 * x * x / sd)
    g /= g.sum()
    return g, i0, i1+1


def point_density(points, sigma, shape, truncate=4.0):
    m, n = shape
    a = np.zeros(shape)
    for y, x in points:
        gy, r0, r1 = gauss1d(y, sigma, truncate, 0, m)
        gx, c0, c1 = gauss1d(x, sigma, truncate, 0, n)
        g = np.outer(gy, gx)
        a[r0:r1, c0:c1] += g
    return a
