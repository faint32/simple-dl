#!/usr/bin/python
# coding: utf-8

"""
    Author: YuJun
    Email: cuteuy@gmail.com
    Date created: 2017/1/20
"""

import random
import numpy as np
import scipy.io


def normalize_data(images):
    # Remove mean of dataset
    mean = images.mean(axis=0)
    images = images - mean

    # Truncate to +/- 3 standard deviations and scale to -1 and +1
    pstd = 3 * images.std()
    images = np.maximum(np.minimum(images, pstd), -pstd) / pstd

    # Rescale from [-1,+1] to [0.1,0.9]
    images = (1 + images) * 0.4 + 0.1

    return images


# 返回已经处理完毕的10000块，与SAE_apply2 load_data功能一致
def sample_images():
    patch_size = 8
    num_patches = 10000
    num_images = 10
    image_size = 512

    image_data = scipy.io.loadmat('data/IMAGES.mat')['IMAGES']

    # Initialize patches with zeros.
    patches = np.zeros(shape=(patch_size * patch_size, num_patches))

    for i in range(num_patches):
        image_id = random.randint(0, num_images - 1)
        image_x = random.randint(0, image_size - patch_size)
        image_y = random.randint(0, image_size - patch_size)

        img = image_data[:, :, image_id]
        patch = img[image_x:image_x + patch_size, image_y:image_y + patch_size].reshape(patch_size * patch_size)
        patches[:, i] = patch

    return normalize_data(patches)


# 返回 10000 份未白化的原始块
def sample_images_raw():
    # import os
    # print os.getcwd()
    image_data = scipy.io.loadmat('data/IMAGES_RAW.mat')['IMAGESr']

    patch_size = 12
    num_patches = 10000
    num_images = image_data.shape[2]
    image_size = image_data.shape[0]

    patches = np.zeros(shape=(patch_size * patch_size, num_patches))

    for i in range(num_patches):
        image_id = random.randint(0, num_images - 1)
        image_x = random.randint(0, image_size - patch_size)
        image_y = random.randint(0, image_size - patch_size)

        img = image_data[:, :, image_id]
        patch = img[image_x:image_x + patch_size, image_y:image_y + patch_size].reshape(patch_size * patch_size)
        patches[:, i] = patch

    return patches
