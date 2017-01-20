#!/usr/bin/python
# coding: utf-8

"""
    Author: YuJun
    Email: cuteuy@gmail.com
    Date created: 2017/1/20
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


def display_color_network(A):
    """
    # display receptive field(s) or basis vector(s) for image patches
    #
    # A         the basis, with patches as column vectors
    # In case the midpoint is not set at 0, we shift it dynamically
    :param A:
    :param file:
    :return:
    """
    if np.min(A) >= 0:
        A -= np.mean(A)

    cols = np.round(np.sqrt(A.shape[1]))

    channel_size = A.shape[0] / 3
    dim = np.sqrt(channel_size)
    dimp = dim + 1
    rows = np.ceil(A.shape[1] / cols)

    B = A[0:channel_size, :]
    C = A[channel_size:2 * channel_size, :]
    D = A[2 * channel_size:3 * channel_size, :]

    B /= np.max(np.abs(B))
    C /= np.max(np.abs(C))
    D /= np.max(np.abs(D))

    # Initialization of the image
    image = np.ones(shape=(dim * rows + rows - 1, dim * cols + cols - 1, 3))

    for i in range(int(rows)):
        for j in range(int(cols)):
            # This sets the patch
            image[i * dimp:i * dimp + dim, j * dimp:j * dimp + dim, 0] = B[:, i * cols + j].reshape(dim, dim)
            image[i * dimp:i * dimp + dim, j * dimp:j * dimp + dim, 1] = C[:, i * cols + j].reshape(dim, dim)
            image[i * dimp:i * dimp + dim, j * dimp:j * dimp + dim, 2] = D[:, i * cols + j].reshape(dim, dim)

    image = (image + 1) / 2

    # PIL.Image.fromarray(np.uint8(image * 255), 'RGB').save(filename)
    return image


def normalize_data(images):
    # Remove mean of dataset
    mean = images.mean(axis=0)
    images = images - mean

    # Truncate to +/- 3 standard deviations and scale to -1 and +1
    pstd = 3 * images.std()
    images = np.maximum(np.minimum(images, pstd), -pstd) / pstd

    # Rescale from [-1,+1] to [0.1,0.9]
    images = (1 + images) * 0.4 + 0.1

    return images


# 返回已经处理完毕的10000块，与SAE_apply2 load_data功能一致
def sample_images():
    patch_size = 8
    num_patches = 10000
    num_images = 10
    image_size = 512

    image_data = scipy.io.loadmat('data/IMAGES.mat')['IMAGES']

    # Initialize patches with zeros.
    patches = np.zeros(shape=(patch_size * patch_size, num_patches))

    for i in range(num_patches):
        image_id = random.randint(0, num_images - 1)
        image_x = random.randint(0, image_size - patch_size)
        image_y = random.randint(0, image_size - patch_size)

        img = image_data[:, :, image_id]
        patch = img[image_x:image_x + patch_size, image_y:image_y + patch_size].reshape(patch_size * patch_size)
        patches[:, i] = patch

    return normalize_data(patches)


# 返回 10000 份未白化的原始块
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
