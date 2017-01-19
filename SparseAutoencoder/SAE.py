#!/usr/bin/python
# coding: utf-8

"""
    Author: YuJun
    Email: cuteuy@gmail.com
    Date created: 2017/1/19
"""
import math
import time
import numpy as np


class SparseAutoencoder(object):
    """
        稀疏自编码：在较多的隐藏单元中，加入系数判罚，表现稀疏性质。
        模拟人类神经系统在某一刺激下，大部分神经元是被抑制的。稀疏意味着系统在尝试去特征选择，找出大量维度中真正重要的若干维。

        自编码：输入单元个数和输出单元个数相同
    """
    def __init__(self, visible_size, hidden_size, rho, lamda, beta):

        self.visible_size = visible_size  # 输入节点个数（不包括偏置节点）
        self.hidden_size = hidden_size  # 隐藏元节点个数
        self.rho = rho  # 稀疏性参数
        self.lamda = lamda  # 权重衰减参数
        self.beta = beta  # 稀疏性惩罚因子权重

        # theta取值边界
        self.limit0 = 0
        self.limit1 = self.limit0 + hidden_size * visible_size
        self.limit2 = self.limit1 + hidden_size * visible_size
        self.limit3 = self.limit2 + hidden_size
        self.limit4 = self.limit3 + visible_size

        # 正太分布初始化各节点权重值
        r = math.sqrt(6) / math.sqrt(visible_size + hidden_size + 1)
        rand = np.random.RandomState(int(time.time()))
        w1 = np.asarray(rand.uniform(low=-r, high=r, size=(hidden_size, visible_size)))
        w2 = np.asarray(rand.uniform(low=-r, high=r, size=(visible_size, hidden_size)))
        # 初始化偏置值为0
        b1 = np.zeros((hidden_size, 1))
        b2 = np.zeros((visible_size, 1))

        # 连接所有参数，作为theta参数值
        self.theta = np.concatenate((w1.flatten(), w2.flatten(), b1.flatten(), b2.flatten()))

    # 激活函数
    def sigmoid(self, x):
        """
        激活函数：把“激活的神经元的特征”通过函数把特征保留并映射出来，这是神经网络能解决非线性问题关键。

        tanh　　　双切正切函数，取值范围[-1,1]
        sigmoid　 采用S形函数，取值范围[0,1]
        ReLU 简单而粗暴，大于0的留下，否则一律为0。

        神经网络中，运算特征是不断进行循环计算，所以在每代循环过程中，每个神经元的值也是在不断变化的。
        这就导致了tanh特征相差明显时的效果会很好，在循环过程中会不断扩大特征效果显示出来。

        但有时候，特征相差比较复杂或是相差不是特别大时，需要更细微的分类判断的时候，sigmoid效果就好了。

        所以sigmoid相比用得更多，但近年发现数据有一个很有意思的特征。也就是稀疏性，数据有很多的冗余，而近似程度的最大保留数据特征，可以用大多数元素为0的稀疏矩阵来实现。
        而Relu，它就是取的max(0,x)，因为神经网络是不断反复计算，实际上变成了它在尝试不断试探如何用一个大多数为0的矩阵来尝试表达数据特征，结果因为稀疏特性的存在，反而这种方法变得运算得又快效果又好了。
        所以，据说，目前大多在用max(0,x)来代替sigmod函数了。


        作者：三符
        链接：https://www.zhihu.com/question/22334626/answer/53203202
        来源：知乎
        """
        return 1 / (1 + np.exp(-x))

    # 代价函数
    def loss_value(self, theta, x_in):
        """
        :type x_in: np.ndarray
        :type theta: np.ndarray
        """
        # 将theta参数分配给具体参数
        w1 = theta[self.limit0: self.limit1].reshape(self.hidden_size, self.visible_size)
        w2 = theta[self.limit1: self.limit2].reshape(self.visible_size, self.hidden_size)
        b1 = theta[self.limit2: self.limit3].reshape(self.hidden_size, 1)
        b2 = theta[self.limit3: self.limit4].reshape(self.visible_size, 1)

        # 正向传导求输出
        hidden_layer = self.sigmoid(np.dot(w1, x_in) + b1)
        output_layer = self.sigmoid(np.dot(w2, hidden_layer) + b2)

        # 计算所有样本的隐藏节点激活值的平均值，以待计算相对熵
        rho_cap = np.sum(hidden_layer, axis=1) / x_in.shape[1]

        # 反向传递
        diff = output_layer - x_in
        sum_of_squares_error = 0.5 * np.sum(np.multiply(diff, diff)) / x_in.shape[1]  # 均方差项
        weight_decay = 0.5 * self.lamda * (np.sum(np.multiply(w1, w1)) + np.sum(np.multiply(w2, w2)))  # 权重衰减项

        kl_divergence = self.beta * np.sum(
            self.rho * np.log(self.rho / rho_cap) + (1 - self.rho) * np.log((1 - self.rho) / (1 - rho_cap))
        )
        cost = sum_of_squares_error + weight_decay + kl_divergence

        kl_div_grad = self.beta * (-(self.rho / rho_cap) + ((1 - self.rho) / (1 - rho_cap)))
        del_out = np.multiply(diff, np.multiply(output_layer, 1 - output_layer))
        del_hid = np.multiply(
            np.dot(np.transpose(w2), del_out) + np.transpose(np.matrix(kl_div_grad)),  # 将kl_div_grad化为“列”并相加
            np.multiply(hidden_layer, 1 - hidden_layer)
        )

        w1_grad = np.dot(del_hid, np.transpose(x_in))
        w2_grad = np.dot(del_out, np.transpose(hidden_layer))
        b1_grad = np.sum(del_hid, axis=1)
        b2_grad = np.sum(del_out, axis=1)

        w1_grad = w1_grad / x_in.shape[1] + self.lamda * w1
        w2_grad = w2_grad / x_in.shape[1] + self.lamda * w2
        b1_grad = b1_grad / x_in.shape[1]
        b2_grad = b2_grad / x_in.shape[1]

        # matrix转为array类型
        w1_grad = np.asarray(w1_grad)
        b1_grad = np.asarray(b1_grad)

        # 连接各参数梯度，作为theta梯度
        theta_grad = np.concatenate((w1_grad.flatten(), w2_grad.flatten(),
                                     b1_grad.flatten(), b2_grad.flatten()))

        # 返回loss值和theta梯度
        return [cost, theta_grad]

