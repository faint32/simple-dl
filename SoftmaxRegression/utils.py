#!/usr/bin/python
# coding: utf-8

"""
    ...
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    Author: YuJun <cuteuy@gmail.com>
"""


import struct
import numpy
import array


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
    dataset = numpy.zeros((num_rows*num_cols, num_examples))

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


def load_mnist_labels(file_name):
    """ Loads the image labels from the provided file name """

    # Open the file
    label_file = open(file_name, 'rb')

    # Read header information from the file
    head1 = label_file.read(4)
    head2 = label_file.read(4)

    # Format the header information for useful data
    num_examples = struct.unpack('>I', head2)[0]

    # Initialize data labels as array of zeros
    labels = numpy.zeros((num_examples, 1), dtype=numpy.int)

    # Read the label data
    labels_raw = array.array('b', label_file.read())
    label_file.close()

    # Copy and return the label data
    labels[:, 0] = labels_raw[:]

    return labels
