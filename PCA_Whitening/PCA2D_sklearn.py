#!/usr/bin/python
# coding: utf-8

"""
    ...
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    Author: YuJun <cuteuy@gmail.com>
"""

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 加载数据并可视化
x = np.loadtxt('data/pcaData.txt', dtype=np.float64)  # n_features * n_samples
fig1 = plt.figure()
plt.scatter(x[0, :], x[1, :], s=40, facecolors='none', edgecolors='b')
plt.title('Raw data')

pca = PCA(whiten=False)
pca.fit(x.T)  # 接收shape (n_samples, n_features)

# 打印主成分所占比重
print pca.explained_variance_ratio_[0] / sum(pca.explained_variance_ratio_)

after_pca = pca.transform(x.T).T
fig2 = plt.figure()
plt.scatter(after_pca[0], after_pca[1])
plt.suptitle('x_rot')
plt.show()
