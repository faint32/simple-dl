#!/usr/bin/python
# coding: utf-8

"""
    Author: YuJun
    Email: cuteuy@gmail.com
    Date created: 2017/1/20
"""

import random

import numpy as np
import matplotlib.pyplot as plt
from utils import display_network, sample_images_raw

# Step 0a: 加载图片数据，随机抽取并显示
x = sample_images_raw()  # 144*10000, 144: 12*12
random_sel = random.sample(range(x.shape[1]), 36)
image_x = display_network(x[:, random_sel])
fig1 = plt.figure()
plt.imshow(image_x, cmap=plt.cm.gray)
plt.title('Raw patch images')


# Step 0b: 使各数据的每个特征（144个像素点：144个特征）平均值为0
x -= np.mean(x, axis=1).reshape(-1, 1)


#  Step 1a: 执行 PCA 并获取 xRot
sigma = x.dot(x.T) / x.shape[1]  # 协方差矩阵
U, s, V = np.linalg.svd(sigma)  # 奇异值分解：https://zh.wikipedia.org/wiki/奇异值分解
x_rot = U.T.dot(x)  # 对144个特征（像素）旋转，使旋转后的这些样本之间无相关性，该处以上个方式显示出来无意义。


# Step 1b: 检验PCA结果旋转矩阵：x_rot的协方差矩阵除了主对角线，其它应该接近0
covar = np.cov(x_rot)  # 使用numpy估计协方差矩阵
fig2 = plt.figure()
plt.imshow(covar)  # 因为主对角线值不大，因此结果可能不明显
plt.title('Covariance matrix of x_rot')


#  Step 2: 找出占据前99%的成分的个数 （共有144个成分）
def get_optimal_k(threshold, s):
    r = 0
    total_sum = np.sum(s)
    sum_ev = 0.0  # Sum of eigenvalues
    for i in range(s.size):
        sum_ev += s[i]
        ratio = sum_ev / total_sum
        if ratio > threshold:
            break
        r += 1
    return r

opt_k_99 = get_optimal_k(0.99, s)
print 'Optimal k to retain 99% variance is:', opt_k_99

# Step 3: 应用PCA进行降维
x_tilde = x_rot[0:opt_k_99, :]  # 保留前k个旋转后的成分
x_hat = U[:, 0:opt_k_99].dot(x_tilde)  # 将旋转后的成分还原
image_x_hat_99 = display_network(x_hat[:, random_sel])  # 展示PCA降维之后的图片

opt_k_90 = get_optimal_k(0.90, s)
print 'Optimal k to retain 90% variance is:', opt_k_90
x_tilde = x_rot[0:opt_k_90, :]
x_hat = U[:, 0:opt_k_90].dot(x_tilde)
image_x_hat_90 = display_network(x_hat[:, random_sel])  # 展示前90%成分效果

f1, ax = plt.subplots(1, 3)
ax[0].imshow(image_x, cmap=plt.cm.gray)
ax[0].set_title('Raw data')
ax[1].imshow(image_x_hat_99, cmap=plt.cm.gray)
ax[1].set_title('99% variance')
ax[2].imshow(image_x_hat_90, cmap=plt.cm.gray)
ax[2].set_title('90% variance')


# Step 4a: PCA白化
epsilon = 0.1  # 正则常数, 图像平滑化，并防止溢出可能
x_PCA_white = np.diag(1.0/np.sqrt(s + epsilon)).dot(x_rot)


# Step 4b: 检验PCA白化：
#     不带正则常数，白化后的矩阵的协方差矩阵，主对角线元素均为1
#     带有正则常数，因为正则常数占比重越来越大，主对角线元素从1逐渐变为0
x_PCA_white_without_regulation = np.diag(1.0/np.sqrt(s)).dot(x_rot)

covar = np.cov(x_PCA_white)
covar_without_regulation = np.cov(x_PCA_white_without_regulation)

f2, ax = plt.subplots(1, 2)
ax[0].imshow(covar)
ax[0].set_title('PCA white With Regulation')
ax[1].imshow(covar_without_regulation)
ax[1].set_title('PCA white Without Regulation')


# Step 5: ZCA白化应用，并可视化处理结果（PCA白化中是各特征的融合，无法可视化，
# ZCA之后便可看出PCA隐式地做了些什么），会发现边缘被增强。

epsilon = 0.1  # Regulation
x_PCA_white = np.diag(1.0/np.sqrt(s + epsilon)).dot(x_rot)
x_ZCA_white = U.dot(x_PCA_white)

image_raw = display_network(x[:, random_sel])
image_ZCA_white = display_network(x_ZCA_white[:, random_sel])

f3, ax = plt.subplots(1, 2)
ax[0].imshow(image_ZCA_white, cmap=plt.cm.gray)
ax[0].set_title('ZCA whitened images')
ax[1].imshow(image_raw, cmap=plt.cm.gray)
ax[1].set_title('Raw images')
plt.show()
