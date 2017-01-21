#!/usr/bin/python
# coding: utf-8

"""
    ...
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    Author: YuJun <cuteuy@gmail.com>
"""

import numpy as np
from PIL import Image


def check_gradient(cost_func, init_theta, first_grad):
    """
    梯度检验：将计算出的梯度值与此处预估的梯度值进行比较。
    :param cost_func: 代价函数
    :param init_theta: 初始化参数
    :param first_grad: 参数梯度
    :return:
    """
    check_epsilon = 0.0001

    num_grad = np.zeros(init_theta.shape)
    for i in range(init_theta.shape[0]):
        theta_epsilon_plus = np.array(init_theta, dtype=np.float64)
        theta_epsilon_plus[i] = init_theta[i] + check_epsilon
        theta_epsilon_minus = np.array(init_theta, dtype=np.float64)
        theta_epsilon_minus[i] = init_theta[i] - check_epsilon

        num_grad[i] = (cost_func(theta_epsilon_plus)[0] - cost_func(theta_epsilon_minus)[0]) / (2 * check_epsilon)
        if i % 100 == 0:
            print "Computing gradient for input:", i

    diff = np.linalg.norm(num_grad - first_grad) / np.linalg.norm(num_grad + first_grad)
    assert diff < 1e-9, 'Difference too large: %s (should be < 1e-9). Check your gradient computation again' % diff


def display_color_network(A, fn):
    """
    # display receptive field(s) or basis vector(s) for image patches
    #
    # A         the basis, with patches as column vectors
    # In case the midpoint is not set at 0, we shift it dynamically
    :param A:
    :param fn:
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

    Image.fromarray(np.uint8(image * 255), 'RGB').save('Image/%s' % fn)
    return image
