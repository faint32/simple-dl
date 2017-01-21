#!/usr/bin/python
# coding: utf-8

"""
    ...
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    Author: YuJun <cuteuy@gmail.com>
"""

import random
import numpy as np
import scipy.io


def display_network(A, opt_normalize=True, opt_graycolor=True, opt_colmajor=True):
    """
    This function visualizes filters in matrix A.

    Parameters
    ----------
    A : Each column of A is a filter.
        We will reshape each column into a square image
        and visualizes on each cell of the visualization panel.
    opt_normalize : whether we need to normalize the filter
        so that all of them can have similar contrast.
    opt_graycolor : whether we use gray as the heat map.
    opt_colmajor: you can switch convention to row major for A.
        In that case, each row of A is a filter. Default value is false.
    """
    # Rescale
    A -= np.average(A)

    # Compute rows & cols
    (row, col) = A.shape
    sz = int(np.ceil(np.sqrt(row)))
    buf = 1
    n = np.ceil(np.sqrt(col))
    m = np.ceil(col / n)

    image = np.ones(shape=(buf + m * (sz + buf), buf + n * (sz + buf)))

    if not opt_graycolor:
        image *= 0.1

    k = 0
    for i in range(int(m)):
        for j in range(int(n)):
            if k >= col:
                continue

            clim = np.max(np.abs(A[:, k]))

            if opt_normalize:
                image[buf + i * (sz + buf):buf + i * (sz + buf) + sz, buf + j * (sz + buf):buf + j * (sz + buf) + sz] = \
                    A[:, k].reshape(sz, sz) / clim
            else:
                image[buf + i * (sz + buf):buf + i * (sz + buf) + sz, buf + j * (sz + buf):buf + j * (sz + buf) + sz] = \
                    A[:, k].reshape(sz, sz) / np.max(np.abs(A))
            k += 1

    return image


# 返回 10000 份原始块
def sample_images_raw():
    # import os
    # print os.getcwd()
    image_data = scipy.io.loadmat('data/IMAGES_RAW.mat')['IMAGESr']

    patch_size = 12
    num_patches = 10000
    num_images = image_data.shape[2]
    image_size = image_data.shape[0]

    patches = np.zeros(shape=(patch_size * patch_size, num_patches))

    for i in range(num_patches):
        image_id = random.randint(0, num_images - 1)
        image_x = random.randint(0, image_size - patch_size)
        image_y = random.randint(0, image_size - patch_size)

        img = image_data[:, :, image_id]
        patch = img[image_x:image_x + patch_size, image_y:image_y + patch_size].reshape(patch_size * patch_size)
        patches[:, i] = patch

    return patches
