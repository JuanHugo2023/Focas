import numpy as np

class Rect(object):
    def __init__(self, row_min, row_max, col_min, col_max, transposed=False):
        self.row_min = row_min
        self.row_max = row_max
        self.col_min = col_min
        self.col_max = col_max
        self.transposed = transposed

    def transform(self, points):
        points = points.copy()
        row, col = points[:, 0], points[:, 1]
        if self.transposed:
            row, col = col, row
        row -= self.row_min
        col -= self.col_min
        points[:, 0], points[:, 1] = row, col
        return points

    def height(self):
        return self.row_max - self.row_min

    def width(self):
        return self.col_max - self.col_min

    def shape(self):
        return self.height(), self.width()

    def reshape(self, new_shape):
        h, w = new_shape
        dh = h - self.height()
        dw = w - self.width()
        return Rect(self.row_min - dh // 2,
                    self.row_max + dh - dh // 2,
                    self.col_min - dw // 2,
                    self.col_max + dw - dw // 2,
                    self.transposed)

    def __contains__(self, point):
        r, c = point
        return self.row_min <= r and r < self.row_max and \
            self.col_min <= c and c < self.col_max

    def __iter__(self):
        return (self.row_min, self.row_max,
                self.col_min, self.col_max).__iter__()

    def __repr__(self):
        if self.transposed:
            transposed_str = ", True"
        else:
            transposed_str = ""
        return "Rect({}, {}, {}, {}{})".format(self.row_min,
                                               self.row_max,
                                               self.col_min,
                                               self.col_max,
                                               transposed_str)
