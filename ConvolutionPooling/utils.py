#!/usr/bin/python
# coding: utf-8

"""
    ...
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    Author: YuJun <cuteuy@gmail.com>
"""

import numpy as np
import scipy.io
import scipy.signal
import scipy.optimize
from sklearn.externals import joblib


def load_wb():
    """
    整合预处理参数：均值化、ZCA白化、线性解码的自编码所有参数整合在一起
    :return: 参数：这些参数能够ZCA白化 8*8*3的RGB图片并将其编码至400个隐藏单元中。
    """
    params = joblib.load('../LinearDecoder/model/ALL_params.pkl')
    opt_param = params['opt_theta']
    zca_white = params['zca_white']
    mean_patch = params['mean_patch']

    visible_size = 8 * 8 * 3  # 稀疏自编码中的输入单元数
    hidden_size = 400  # 稀疏自编码中的隐藏单元数

    limit0 = 0
    limit1 = hidden_size * visible_size
    limit2 = 2 * hidden_size * visible_size
    limit3 = 2 * hidden_size * visible_size + hidden_size

    opt_w1 = opt_param[limit0: limit1].reshape(hidden_size, visible_size)
    opt_b1 = opt_param[limit2: limit3].reshape(hidden_size, 1)

    W = np.dot(opt_w1, zca_white)
    b = opt_b1 - np.dot(W, mean_patch)
    return W, b


def load_training_dataset():
    """ Loads the images and labels as np arrays
        The dataset is originally read as a dictionary """

    train_data = scipy.io.loadmat('data/stlTrainSubset.mat')
    train_images = np.array(train_data['trainImages'])
    train_labels = np.array(train_data['trainLabels'])

    return [train_images, train_labels]


def load_test_dataset():
    """ Loads the images and labels as np arrays
        The dataset is originally read as a dictionary """

    test_data = scipy.io.loadmat('data/stlTestSubset.mat')
    test_images = np.array(test_data['testImages'])
    test_labels = np.array(test_data['testLabels'])

    return [test_images, test_labels]


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    train_images, train_labels = load_training_dataset()
    for i in range(100):
        plt.imshow(train_images[:, :, :, i], interpolation='nearest')
        cat = train_labels[i][0]
        if cat == 1:
            print '飞机'
        elif cat == 2:
            print '汽车'
        elif cat == 3:
            print '猫科动物'
        else:
            print '犬科动物'
        plt.show()
