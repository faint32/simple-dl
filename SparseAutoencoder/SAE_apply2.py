#!/usr/bin/python
# coding: utf-8

"""
    Author: YuJun
    Email: cuteuy@gmail.com
    Date created: 2017/1/19
"""

import array
import struct
import scipy.io
import numpy as np
import scipy.optimize
import matplotlib.pyplot
from SAE import SparseAutoencoder


def load_mnist_images(file_name):
    """ Loads the images from the provided file name """

    image_file = open(file_name, 'rb')  # Open the file

    # Read header information from the file
    head1 = image_file.read(4)
    head2 = image_file.read(4)
    head3 = image_file.read(4)
    head4 = image_file.read(4)

    # Format the header information for useful data
    num_examples = struct.unpack('>I', head2)[0]
    num_rows = struct.unpack('>I', head3)[0]
    num_cols = struct.unpack('>I', head4)[0]

    # Initialize dataset as array of zeros
    dataset = np.zeros((num_rows*num_cols, num_examples))

    # Read the actual image data
    images_raw = array.array('B', image_file.read())
    image_file.close()

    # Arrange the data in columns
    for i in range(num_examples):

        limit1 = num_rows * num_cols * i
        limit2 = num_rows * num_cols * (i + 1)

        dataset[:, i] = images_raw[limit1: limit2]

    # Normalize and return the dataset
    return dataset / 255


def visualize_w1(opt_w1, vis_side, hid_side):
    """ Add the weights as a matrix of images """

    figure, axes = matplotlib.pyplot.subplots(nrows=hid_side,
                                              ncols=hid_side)
    index = 0

    for axis in axes.flat:
        """ Add row of weights as an image to the plot """

        image = axis.imshow(opt_w1[index, :].reshape(vis_side, vis_side),
                            cmap=matplotlib.pyplot.cm.gray, interpolation='nearest')
        axis.set_frame_on(False)
        axis.set_axis_off()
        index += 1

    """ Show the obtained plot """

    matplotlib.pyplot.show()


def execute():
    vis_side = 28
    hid_side = 8
    rho = 0.1  # desired average activation of hidden units
    lamda = 0.001  # weight decay parameter
    beta = 3  # weight of sparsity penalty term
    num_pics = 2000  # number of training examples
    max_iterations = 400  # number of optimization iterations

    visible_size = vis_side * vis_side  # number of input units
    hidden_size = hid_side * hid_side  # number of hidden units

    training_data = load_mnist_images('data/train-images.idx3-ubyte')[:, :num_pics]
    from scipy.misc import toimage
    toimage(training_data[:, 1].reshape((28, 28))).show()

    encoder = SparseAutoencoder(visible_size, hidden_size, rho, lamda, beta)

    print '...Training'
    opt_solution = scipy.optimize.minimize(
        encoder.loss_value, encoder.theta, args=(training_data,), method='L-BFGS-B',
        jac=True, options={'maxiter': max_iterations}
    )
    print 'Train over.'
    opt_theta = opt_solution.x
    opt_w1 = opt_theta[encoder.limit0: encoder.limit1].reshape(hidden_size, visible_size)

    visualize_w1(opt_w1, vis_side, hid_side)


execute()
