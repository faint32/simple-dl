#!/usr/bin/python
# coding: utf-8

"""
    卷积神经网络：
        卷积层：
            卷积核来自线性解码的自编码器，自编码器中，共有400个隐藏层，因此这里设置400个卷积核。
            对应的卷积层中有400个通道（相对于输入层有RGB三个通道），每个通道内部权重共享进行卷积。
        池化层：
            使用大小为3的池进行池化，池化时取平均值。
        SoftMax回归层。

        这里再进行训练的时候，反向传导仅训练了最后一层，而卷积层和池化层并未进行微调，直接使用的自编码时的权重。
        用于分辨：猫科动物、犬科动物、汽车、飞行器，准确率为

    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    Author: YuJun <cuteuy@gmail.com>
"""
import scipy.io
import numpy as np
import scipy.signal
import scipy.optimize
from sklearn.externals import joblib
from SoftmaxRegression.SR import SoftmaxRegression
from utils import load_test_dataset, load_training_dataset, load_wb


class CNN(object):
    def __init__(self, w, b, kernel_size, pool_size):
        """
        :param w: kernel核参数W
        :param b: kernel核参数b
        :param kernel_size: 卷积核边长
        :param pool_size:  池化边长
        """
        self.W = w
        self.b = b
        self.kernel_size = kernel_size
        self.pool_size = pool_size

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def convolve(self, input_images, num_features):
        image_dim = input_images.shape[0]
        image_channels = input_images.shape[2]
        num_images = input_images.shape[3]

        conv_dim = image_dim - self.kernel_size + 1  # 卷积层中每层的边长
        convolved_features = np.zeros((num_features, num_images, conv_dim, conv_dim))  # 分配存储卷积结果的内存

        for image_num in range(num_images):

            for feature_num in range(num_features):

                # Initialize convolved image as array of zeros
                convolved_image = np.zeros((conv_dim, conv_dim))

                for channel in range(image_channels):
                    # 获取卷积核
                    limit0 = self.kernel_size * self.kernel_size * channel
                    limit1 = limit0 + self.kernel_size * self.kernel_size
                    feature = self.W[feature_num, limit0: limit1].reshape(self.kernel_size, self.kernel_size)

                    image = input_images[:, :, channel, image_num]  # 待卷积的某个通道的图片
                    convolved_image += scipy.signal.convolve2d(image, feature, 'valid')  # 卷积

                # 激活卷积层，并添加至最终卷积的结果中
                convolved_image = self.sigmoid(convolved_image + self.b[feature_num, 0])
                convolved_features[feature_num, image_num, :, :] = convolved_image

        return convolved_features

    def pool(self, convolved_features):
        num_features = convolved_features.shape[0]
        num_images = convolved_features.shape[1]
        conv_dim = convolved_features.shape[2]
        final_dim = conv_dim / self.pool_size

        pooled_features = np.zeros((num_features, num_images, final_dim, final_dim))

        for image_num in range(num_images):
            for feature_num in range(num_features):
                for pool_row in range(final_dim):
                    row_start = pool_row * self.pool_size
                    row_end = row_start + self.pool_size

                    for pool_col in range(final_dim):
                        col_start = pool_col * self.pool_size
                        col_end = col_start + self.pool_size
                        # 求平均并池化
                        patch = convolved_features[feature_num, image_num, row_start: row_end, col_start: col_end]
                        pooled_features[feature_num, image_num, pool_row, pool_col] = np.mean(patch)

        return pooled_features


def get_pooled_features(network, images, num_features, step_size):
    """
    :type network: CNN
    :param network: 初始化的cnn模型
    :param images: shape为(边长，边长，通道数，图片数)的图片数据
    :param num_features: 卷积核个数
    :param step_size: 每步处理的patch个数
    :return:
    """
    kernel_size = network.kernel_size
    pool_size = network.pool_size
    img_size = images.shape[0]
    num_images = images.shape[3]

    final_dim = (img_size-kernel_size+1) / pool_size

    pooled_features_data = np.zeros((num_features, num_images, final_dim, final_dim))  # 申请容器

    for step in range(num_images / step_size):
        # 因为卷积/池化时额外需要大量内存，故分多步骤进行
        limit0 = step_size * step
        limit1 = step_size * (step+1)
        image_batch = images[:, :, :, limit0: limit1]

        convolved_features = network.convolve(image_batch, num_features)
        pooled_features = network.pool(convolved_features)

        pooled_features_data[:, limit0: limit1, :, :] = pooled_features

        # 防止内存溢出
        del image_batch
        del convolved_features
        del pooled_features
        print 'Step: %s of %s' % (step + 1, num_images / step_size)

    # 数据转维，以便训练和测试
    input_size = pooled_features_data.size / num_images
    # num_features, num_images, dim, dim --> num_features, dim, dim, num_images
    pooled_features_data = np.transpose(pooled_features_data, (0, 2, 3, 1))
    # input_size <- num_features * dim * dim
    pooled_features_data = pooled_features_data.reshape(input_size, num_images)
    return pooled_features_data


def execute_cnn():
    # 卷积神经网络参数
    kernel_size = 8  # 这里使用的kernel就是自编码的，与训练自编码时的patch size一致
    pool_size = 3  # 池大小
    step_size = 50  # 每步获取50张图片经卷积——池化后的结果，防止内存溢出
    hidden_size = 400  # 卷积核个数，与自编码中编码后的隐藏层个数一致
    opt_w, opt_b = load_wb()

    cnn = CNN(opt_w, opt_b, kernel_size, pool_size)

    print '... Loading data'
    train_images, train_labels = load_training_dataset()
    test_images, test_labels = load_test_dataset()
    train_labels -= 1   # map class from [1, 2, 3, 4] to [0, 1, 2, 3]
    test_labels -= 1    # map class from [1, 2, 3, 4] to [0, 1, 2, 3]

    print '... Process data(convolve and pool)'
    train_data = get_pooled_features(cnn, train_images, hidden_size, step_size)
    test_data = get_pooled_features(cnn, test_images, hidden_size, step_size)
    # 这一步花费时间太多时间，所以将加载后的数据保存一下，下次运行可直接调用。
    try:
        np.save('data/ProcessedTrainSet.npy', train_data)
        np.save('data/ProcessedTestSet.npy', test_data)
    except Exception as e:
        print e.message

    # SoftMax回归参数
    input_size = train_data.shape[0]
    num_classes = 4
    lambda_ = 0.0001

    print '... Training'
    regressor = SoftmaxRegression(input_size, num_classes, lambda_)
    opt_solution = scipy.optimize.minimize(
        regressor.softmax_cost, regressor.theta, args=(train_data, train_labels,),
        method='L-BFGS-B', jac=True, options={'maxiter': 500, 'disp': True}
    )
    opt_theta = opt_solution.x
    joblib.dump(opt_theta, 'model/CNN_SR_theta.pkl')

    print '... Testing'
    opt_theta = joblib.load('model/CNN_SR_theta.pkl')
    predictions = regressor.softmax_predict(opt_theta, test_data)
    correct = test_labels[:, 0] == predictions[:, 0]
    print 'Accuracy :', np.mean(correct)


execute_cnn()
