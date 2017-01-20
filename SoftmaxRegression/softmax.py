#!/usr/bin/python
# coding: utf-8

"""
    Author: YuJun
    Email: cuteuy@gmail.com
    Date created: 2017/1/20
"""

import time
import numpy as np
import scipy.sparse
import scipy.optimize
from utils import load_mnist_images, load_mnist_labels


class SoftmaxRegression(object):
    """ The Softmax Regression class """

    def __init__(self, input_size, num_classes, lamda):
        self.input_size = input_size    # input vector size
        self.num_classes = num_classes  # number of classes
        self.lamda = lamda              # weight decay parameter

        # Randomly initialize the class weights
        rand = np.random.RandomState(int(time.time()))
        self.theta = 0.005 * np.asarray(rand.normal(size=(num_classes*input_size, 1)))

    def get_groundtruth(self, labels):
        labels = labels.flatten()
        data = np.ones(len(labels))
        indptr = np.arange(len(labels)+1)

        # 将60000个标签转化为 10 * 60000 的矩阵，其中每一列代表每个样本的label，值为1的索引值表示该label值
        ground_truth = scipy.sparse.csr_matrix((data, labels, indptr))
        ground_truth = ground_truth.todense().transpose()
        return ground_truth

    def softmax_cost(self, theta, x_in, x_labels):
        ground_truth = self.get_groundtruth(x_labels)

        # 每个类别有一组theta值，一组theta有像素数量的个数。
        theta = theta.reshape(self.num_classes, self.input_size)

        # 计算每个样本属于各类的几率
        theta_x = np.dot(theta, x_in)  # dot(n_class * input_size, input_size * n_sample) = n_class * n_sample
        hypothesis = np.exp(theta_x)
        probabilities = hypothesis / np.sum(hypothesis, axis=0)

        # 保留应该正确的那个类别的实际预测几率，并计算其平均损失值
        cost_examples = np.multiply(ground_truth, np.log(probabilities))
        traditional_cost = -(np.sum(cost_examples) / x_in.shape[1])

        # 计算权重衰减项的代价：加入该数，使得代价函数为严格凸函数，有唯一解，防止在取多解时的某个解时，可能造成过拟合
        theta_squared = np.multiply(theta, theta)
        weight_decay = 0.5 * self.lamda * np.sum(theta_squared)

        cost = traditional_cost + weight_decay

        theta_grad = -np.dot(ground_truth - probabilities, np.transpose(x_in))
        theta_grad = theta_grad / x_in.shape[1] + self.lamda * theta
        theta_grad = np.array(theta_grad)
        theta_grad = theta_grad.flatten()

        return [cost, theta_grad]

    def softmax_predict(self, theta, x_in):
        """Returns predicted classes for a set of inputs"""

        # Reshape 'theta' for ease of computation
        theta = theta.reshape(self.num_classes, self.input_size)

        # Compute the class probabilities for each example
        theta_x = np.dot(theta, x_in)
        hypothesis = np.exp(theta_x)
        probabilities = hypothesis / np.sum(hypothesis, axis=0)

        # Give the predictions based on probability values
        predictions = np.zeros((x_in.shape[1], 1))
        predictions[:, 0] = np.argmax(probabilities, axis=0)

        return predictions

    @staticmethod
    def predict_out(theta, x_in, num_classes, input_size):
        """Returns predicted classes for a set of inputs"""

        # Reshape 'theta' for ease of computation
        theta = theta.reshape(num_classes, input_size)

        # Compute the class probabilities for each example
        theta_x = np.dot(theta, x_in)
        hypothesis = np.exp(theta_x)
        probabilities = hypothesis / np.sum(hypothesis, axis=0)

        # Give the predictions based on probability values
        predictions = np.zeros((x_in.shape[1], 1))
        predictions[:, 0] = np.argmax(probabilities, axis=0)

        return predictions


def execute():
    """ Loads data, trains the model and predicts classes for test data """
    input_size = 784
    num_classes = 10
    lamda = 0.0001
    max_iterations = 100

    training_data = load_mnist_images('data/train-images.idx3-ubyte')  # 784 * 60000
    training_labels = load_mnist_labels('data/train-labels.idx1-ubyte')  # 60000 * 1

    regressor = SoftmaxRegression(input_size, num_classes, lamda)

    print '... Training'
    opt_solution = scipy.optimize.minimize(
        regressor.softmax_cost, regressor.theta,
        args=(training_data, training_labels,), method='L-BFGS-B',
        jac=True, options={'maxiter': max_iterations, 'disp': True}
    )
    opt_theta = opt_solution.x

    test_data = load_mnist_images('data/t10k-images.idx3-ubyte')
    test_labels = load_mnist_labels('data/t10k-labels.idx1-ubyte')

    print '... Testing'
    predictions = regressor.softmax_predict(opt_theta, test_data)

    correct = test_labels[:, 0] == predictions[:, 0]
    print "Accuracy :", np.mean(correct)


if __name__ == '__main__':
    execute()
