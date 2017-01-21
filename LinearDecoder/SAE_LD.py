#!/usr/bin/python
# coding: utf-8

"""
    输出层使用线性激活函数的稀疏自编码器
    1、ZCA白化
    2、白化后的图片进行 线性解码的稀疏自编码。
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    Author: YuJun <cuteuy@gmail.com>
"""

# coding: utf-8
import time
import numpy as np
import scipy.io
import scipy.optimize
from sklearn.externals import joblib
from utils import display_color_network, check_gradient


class SparseAutoEncoderLD(object):
    def __init__(self, visible_size, hidden_size, rho, lambda_, beta):
        self.visible_size = visible_size
        self.hidden_size = hidden_size
        self.rho = rho
        self.lambda_ = lambda_
        self.beta = beta

        self.limit0 = 0
        self.limit1 = hidden_size * visible_size
        self.limit2 = 2 * hidden_size * visible_size
        self.limit3 = 2 * hidden_size * visible_size + hidden_size
        self.limit4 = 2 * hidden_size * visible_size + hidden_size + visible_size

        r = np.sqrt(6) / np.sqrt(visible_size + hidden_size + 1)

        rand = np.random.RandomState(int(time.time()))

        w1 = np.asarray(rand.uniform(low=-r, high=r, size=(hidden_size, visible_size)))
        w2 = np.asarray(rand.uniform(low=-r, high=r, size=(visible_size, hidden_size)))

        b1 = np.zeros((hidden_size, 1))
        b2 = np.zeros((visible_size, 1))

        self.theta = np.concatenate((w1.flatten(), w2.flatten(), b1.flatten(), b2.flatten()))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def loss_value(self, theta, input_data):

        w1 = theta[self.limit0: self.limit1].reshape(self.hidden_size, self.visible_size)
        w2 = theta[self.limit1: self.limit2].reshape(self.visible_size, self.hidden_size)
        b1 = theta[self.limit2: self.limit3].reshape(self.hidden_size, 1)
        b2 = theta[self.limit3: self.limit4].reshape(self.visible_size, 1)

        hidden_layer = self.sigmoid(np.dot(w1, input_data) + b1)
        output_layer = np.dot(w2, hidden_layer) + b2    # 变化：变为线性解码

        rho_cap = np.sum(hidden_layer, axis=1) / input_data.shape[1]

        diff = output_layer - input_data
        sum_of_squares_error = 0.5 * np.sum(np.multiply(diff, diff)) / input_data.shape[1]
        weight_decay = 0.5 * self.lambda_ * (np.sum(np.multiply(w1, w1)) + np.sum(np.multiply(w2, w2)))
        kl_divergence = self.beta * np.sum(
            self.rho * np.log(self.rho / rho_cap) + (1 - self.rho) * np.log((1 - self.rho) / (1 - rho_cap))
        )
        cost = sum_of_squares_error + weight_decay + kl_divergence

        kl_div_grad = self.beta * (-(self.rho / rho_cap) + ((1 - self.rho) / (1 - rho_cap)))
        del_out = diff      # 变化：解码误差项
        del_hid = np.multiply(
            np.dot(np.transpose(w2), del_out) + np.transpose(np.matrix(kl_div_grad)),
            np.multiply(hidden_layer, 1 - hidden_layer)
        )

        w1_grad = np.dot(del_hid, np.transpose(input_data))
        w2_grad = np.dot(del_out, np.transpose(hidden_layer))
        b1_grad = np.sum(del_hid, axis=1)
        b2_grad = np.sum(del_out, axis=1)

        w1_grad = w1_grad / input_data.shape[1] + self.lambda_ * w1
        w2_grad = w2_grad / input_data.shape[1] + self.lambda_ * w2
        b1_grad = b1_grad / input_data.shape[1]
        b2_grad = b2_grad / input_data.shape[1]

        w1_grad = np.array(w1_grad)
        w2_grad = np.array(w2_grad)
        b1_grad = np.array(b1_grad)
        b2_grad = np.array(b2_grad)

        theta_grad = np.concatenate((w1_grad.flatten(), w2_grad.flatten(),
                                     b1_grad.flatten(), b2_grad.flatten()))

        return [cost, theta_grad]


def exec_task(debug=False):
    # 自编码参数
    sparsity_param = 0.035  # 期望稀疏率
    lambda_ = 3e-3          # 权重衰减比重
    beta = 5                # 稀疏判罚比重

    # 白化参数
    white_epsilon = 0.1  # 正则常数

    if debug:
        print 'Gradient checking...'
        debug_hidden_size = 5
        debug_visible_size = 8
        patches = np.random.rand(8, 10)
        debug_encoder = SparseAutoEncoderLD(debug_visible_size, debug_hidden_size, sparsity_param, lambda_, beta)
        cost, grad = debug_encoder.loss_value(debug_encoder.theta, patches)
        check_gradient(lambda x: debug_encoder.loss_value(x, patches), debug_encoder.theta, grad)

    print 'Loading patches and applying whitening...'
    patches = scipy.io.loadmat('data/stlSampledPatches.mat')['patches']

    # 训练参数
    visible_size = patches.shape[0]  # 输入单元数 为 8*8*3
    hidden_size = 400  # 隐藏单元数

    display_color_network(patches[:, 0:100], fn='patches_raw.png')  # 显示原图片数据

    # ZCA白化
    patch_mean = np.mean(patches, axis=1)
    patches -= patch_mean.reshape((-1, 1))
    sigma = patches.dot(patches.transpose()) / patches.shape[1]
    (U, s, V) = np.linalg.svd(sigma)
    zca_white = U.dot(np.diag(1 / (s + white_epsilon))).dot(U.T)
    patches_zca = zca_white.dot(patches)
    # # 上两行代码等价于下面注释掉的三行，效率更高，且模型易保存
    # patch_rot = U.T.dot(patches)
    # pca_whiten = np.diag(1 / (s + white_epsilon)).dot(patch_rot)
    # zca_whiten = U.dot(pca_whiten)
    display_color_network(patches_zca[:, 0:100], fn='patches_zca.png')

    print 'Running sparse auto encoding with linear decoder...'
    encoder = SparseAutoEncoderLD(visible_size, hidden_size, sparsity_param, lambda_, beta)
    opt_solution = scipy.optimize.minimize(
        encoder.loss_value, encoder.theta, args=(patches_zca,),
        method='L-BFGS-B', jac=True, options={'maxiter': 400, 'disp': True}
    )   # after install Theano, this use multi-core.
    opt_theta = opt_solution.x
    w1 = opt_theta[0:hidden_size * visible_size].reshape(hidden_size, visible_size)
    display_color_network(w1.dot(zca_white).transpose(), 'linear_decoder_features.png')  # weight after zca processing

    # 保存训练参数，以供之后使用
    params = dict({})
    params['opt_theta'] = opt_theta
    params['zca_white'] = zca_white
    params['mean_patch'] = patch_mean
    joblib.dump(params, "model/ALL_params.pkl", compress=3)


if __name__ == '__main__':
    exec_task(debug=True)
