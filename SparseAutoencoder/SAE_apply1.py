#!/usr/bin/python
# coding: utf-8

"""
    Author: YuJun
    Email: cuteuy@gmail.com
    Date created: 2017/1/19
"""

import time
import numpy as np
import scipy.io
import scipy.optimize
import matplotlib.pyplot
from SAE import SparseAutoencoder


def normalize_dataset(dataset):
    # Remove mean of dataset
    dataset -= np.mean(dataset)

    # Truncate to +/-3 standard deviations and scale to -1 to 1
    std_dev = 3 * np.std(dataset, ddof=1)
    dataset = np.maximum(np.minimum(dataset, std_dev), -std_dev) / std_dev

    # Rescale from [-1, 1] to [0.1, 0.9]
    dataset = (dataset + 1) * 0.4 + 0.1

    return dataset


def load_dataset(num_patches, patch_side):
    """ 将num_patches块大小为patch_side*patch_side的图片加载至array并返回 """

    images = scipy.io.loadmat('data/IMAGES.mat')
    images = images['IMAGES']

    from scipy.misc import toimage
    toimage(images[:, :, 1]).show()

    # 初始化容器
    dataset = np.zeros((patch_side * patch_side, num_patches))

    # 初始化块选取未知和截取块的图片选取位置
    rand = np.random.RandomState(int(time.time()))
    image_indices = rand.randint(512 - patch_side, size=(num_patches, 2))
    image_number = rand.randint(10, size=num_patches)

    # 抽样、存储、归一化并返回
    for i in xrange(num_patches):
        index1 = image_indices[i, 0]
        index2 = image_indices[i, 1]
        index3 = image_number[i]

        patch = images[index1:(index1+patch_side), index2:(index2+patch_side), index3]
        patch = patch.flatten()
        dataset[:, i] = patch

    return normalize_dataset(dataset)


# 将转化为稀疏节点的各权重在图片的相应位置展示出来
def visualize_w1(opt_w1, vis_patch_side, hid_patch_side):
    figure, axes = matplotlib.pyplot.subplots(nrows=hid_patch_side, ncols=hid_patch_side)
    index = 0

    for axis in axes.flat:
        # 构成每个中间节点的权重
        image = axis.imshow(opt_w1[index, :].reshape(vis_patch_side, vis_patch_side),
                            cmap=matplotlib.pyplot.cm.gray, interpolation='nearest')
        axis.set_frame_on(False)
        axis.set_axis_off()
        index += 1

    # 越白表示权重值越大
    matplotlib.pyplot.show()


# 执行稀疏自编码
def execute():
    """ Define the parameters of the Autoencoder """

    vis_patch_side = 8  # side length of sampled image patches
    hid_patch_side = 5  # side length of representative image patches
    rho = 0.01  # desired average activation of hidden units
    lamda = 0.0001  # weight decay parameter
    beta = 3  # weight of sparsity penalty term
    num_patches = 10000  # number of training examples
    max_iterations = 400  # number of optimization iterations

    visible_size = vis_patch_side * vis_patch_side  # number of input units
    hidden_size = hid_patch_side * hid_patch_side  # number of hidden units

    training_data = load_dataset(num_patches, vis_patch_side)
    encoder = SparseAutoencoder(visible_size, hidden_size, rho, lamda, beta)

    print '...Training'
    """ Run the L-BFGS algorithm to get the optimal parameter values """
    opt_solution = scipy.optimize.minimize(
        encoder.loss_value, encoder.theta, args=(training_data,), method='L-BFGS-B',
        jac=True, options={'maxiter': max_iterations}
    )
    print 'Train over.'
    opt_theta = opt_solution.x
    opt_w1 = opt_theta[encoder.limit0: encoder.limit1].reshape(hidden_size, visible_size)

    visualize_w1(opt_w1, vis_patch_side, hid_patch_side)


execute()
