#!/usr/bin/python
# coding: utf-8

"""
    Author: YuJun
    Email: cuteuy@gmail.com
    Date created: 2017/1/20
"""
import array
import struct
import numpy as np


def load_mnist_images(file_name):
    """ Loads the images from the provided file name """

    image_file = open(file_name, 'rb')  # Open the file

    # Read header information from the file
    head1 = image_file.read(4)
    head2 = image_file.read(4)
    head3 = image_file.read(4)
    head4 = image_file.read(4)

    # Format the header information for useful data
    num_examples = struct.unpack('>I', head2)[0]
    num_rows = struct.unpack('>I', head3)[0]
    num_cols = struct.unpack('>I', head4)[0]

    # Initialize dataset as array of zeros
    dataset = np.zeros((num_rows*num_cols, num_examples))

    # Read the actual image data
    images_raw = array.array('B', image_file.read())
    image_file.close()

    # Arrange the data in columns
    for i in range(num_examples):

        limit1 = num_rows * num_cols * i
        limit2 = num_rows * num_cols * (i + 1)

        dataset[:, i] = images_raw[limit1: limit2]

    # Normalize and return the dataset
    return dataset / 255


def load_mnist_labels(file_name):
    """ Loads the image labels from the provided file name """

    # Open the file
    label_file = open(file_name, 'rb')

    # Read header information from the file
    head1 = label_file.read(4)
    head2 = label_file.read(4)

    # Format the header information for useful data
    num_examples = struct.unpack('>I', head2)[0]

    # Initialize data labels as array of zeros
    labels = np.zeros((num_examples, 1), dtype=np.int)

    # Read the label data
    labels_raw = array.array('b', label_file.read())
    label_file.close()

    # Copy and return the label data
    labels[:, 0] = labels_raw[:]

    return labels


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
