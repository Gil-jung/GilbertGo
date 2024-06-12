import six.moves.cPickle as pickle
import gzip
import numpy as np


def encode_label(j):  # We one-hot encode indices to vectors of length 10.
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def shape_data(data):
    features = [np.reshape(x, (784, 1)) for x in data[0]]  # We flatten the input images to feature vectors of length 784.
    
    labels = [encode_label(y) for y in data[1]]  # All labels are one-hot encoded.

    return list(zip(features, labels))  # Then we create pairs of features and labels.


def load_data_impl():
    # file retrieved by:
    #   wget https://s3.amazonaws.com/img-datasets/mnist.npz -O code/dlgo/nn/mnist.npz
    # code based on:
    #   site-packages/keras/datasets/mnist.py
    path = './code/dlgo/nn/mnist.npz'
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)

def load_data():
    train_data, test_data = load_data_impl()
    return shape_data(train_data), shape_data(test_data)  # We discard validation data here and reshape the other two data sets.