#!/usr/bin/python
# coding: utf-8

"""
    此处仅有主成分的处理后的数据结果，未可视化。
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    Author: YuJun <cuteuy@gmail.com>
"""

import random
from sklearn.decomposition import PCA
from utils import sample_images_raw

# 加载图片数据
x = sample_images_raw()  # n_features * n_samples
num_samples = x.shape[1]
random_sel = random.sample(range(num_samples), 36)


# 仅保留前80个主成分（公144个成分）
pca = PCA(80)
pca.fit(x.transpose())
patches_reduced = pca.transform(x.transpose()).transpose()
print patches_reduced.shape


# 仅保留前80个主成分（公144个成分），同时保证各成分的方差一致，均为单位方差
pca = PCA(80, whiten=True)
pca.fit(x.transpose())
patches_reduced = pca.transform(x.transpose()).transpose()
print patches_reduced.shape
