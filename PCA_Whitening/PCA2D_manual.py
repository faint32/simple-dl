#!/usr/bin/python
# coding: utf-8

"""
    PCA在2D散点下的效果。
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    Author: YuJun <cuteuy@gmail.com>
"""

import numpy as np
import matplotlib.pyplot as plt


# 加载数据并可视化
x = np.loadtxt('data/pcaData.txt', dtype=np.float64)

fig1 = plt.figure()
plt.scatter(x[0, :], x[1, :], s=40, facecolors='none', edgecolors='b')
plt.title('Raw data')


# 通过SVD获取旋转矩阵（新坐标基）并在原图显示
n, m = x.shape
sigma = x.dot(x.T) / m  # x平均值已为0

U, s, V = np.linalg.svd(sigma)

# 打印主成分所占比重
print s[0] / sum(s)

fig2 = plt.figure()
plt.scatter(x[0, :], x[1, :], s=40, facecolors='none', edgecolors='green')
plt.title('Plot u1 and u2')
plt.plot([0, U[0, 0]], [0, U[1, 0]], color='blue')
plt.plot([0, U[0, 1]], [0, U[1, 1]], color='blue')
plt.xlim([-0.8, 0.8])
plt.ylim([-0.8, 0.8])


x_rot = U.T.dot(x)  # 对x进行旋转

# 绘出旋转后的在新基下的点
fig3 = plt.figure()
plt.scatter(x_rot[0, :], x_rot[1, :], s=40, facecolors='none', edgecolors='blue')
plt.title('x_rot')

# 取前k个主成分，并绘出
k = 1
x_tilde = x_rot[0:k, :]
x_hat = U[:, 0:k].dot(x_tilde)

fig4 = plt.figure()
plt.scatter(x_hat[0, :], x_hat[1, :], s=40, facecolors='none', edgecolors='blue')
plt.title('x_hat')


# Whiten 白化：利用PCA的前面步骤，消除特征之间的关联性。并将使各方向的方差一致。之后绘出白化后的结果。
epsilon = 1e-5  # 对输入图像也有一些平滑(或低通滤波)的作用。这样处理还能消除在图像的像素信息获取过程中产生的噪声，改善学习到的特征
x_PCA_white = np.diag(1.0/np.sqrt(s + epsilon)).dot(x_rot)

fig5 = plt.figure()
plt.scatter(x_PCA_white[0, :], x_PCA_white[1, :], s=40, facecolors='none', edgecolors='blue')
plt.title('x_PCA_white')

# 将白化后的数据再次旋转过去。即ZCA，并绘出结果
x_ZCA_white = U.dot(x_PCA_white)

fig = plt.figure()
plt.scatter(x_ZCA_white[0, :], x_ZCA_white[1, :], s=40, facecolors='none', edgecolors='blue')
plt.title('x_ZCA_white')

# 显示所有
plt.show()
