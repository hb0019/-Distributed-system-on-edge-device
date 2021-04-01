# -*- coding:utf-8 -*-

import pickle
import random
import numpy as np
from numpy.lib.ufunclike import _deprecate_out_named_y

class_num = 9
image_size = 64
img_channels = 1


# ========================================================== #
# ├─ prepare_data()
#  ├─ download training data if not exist by download_data()
#  ├─ load data by load_data()
#  └─ shuffe and return data
# ========================================================== #

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_data_one(file):
    batch = unpickle(file)
    data = batch['data']
    labels = batch['labels']
    print("Loading %s : %d." % (file, len(data)))
    return data, labels


def load_data(files, data_dir, label_count):
    global image_size, img_channels
    data, labels = load_data_one(data_dir + '/' + files[0])
    for f in files[1:]:
        data_n, labels_n = load_data_one(data_dir + '/' + f)
        data = np.append(data, data_n, axis=0)
        labels = np.append(labels, labels_n, axis=0)
    labels = np.array([[float(i == label) for i in range(label_count)] for label in labels])
    data = data.reshape([-1, img_channels, image_size, image_size])
    data = data.transpose([0, 2, 3, 1])
    return data, labels


def load_data_mine(files, data_dir, label_count):
    global image_size, img_channels
    data, labels = load_data_one(data_dir + '/' + files[0])
    for f in files[1:]:
        data_n, labels_n = load_data_one(data_dir + '/' + f)
        data = np.concatenate((data, data_n))
        labels = np.concatenate((labels, labels_n))

    num_data = len(labels)
    labels = np.array([[float(i == label) for i in range(label_count)] for label in labels])
    data = data.reshape((num_data, image_size * image_size, img_channels), order='F')
    data = data.reshape((num_data, image_size, image_size, img_channels))
    return data, labels


def prepare_data():
    print("======Loading data======")
    data_dir = './NEU-CLS-64-python'
    image_dim = image_size * image_size * img_channels
    with open(data_dir + "/batches.meta") as f:
        label_names = f.read().split("\n")

    label_count = len(label_names)
    train_files = ['data_batch_1', 'data_batch_2', 'data_batch_3']
    train_data, train_labels = load_data_mine(train_files, data_dir, label_count)
    test_data, test_labels = load_data_mine(['test_batch'], data_dir, label_count)

    print("Train data:", np.shape(train_data), np.shape(train_labels))
    print("Test data :", np.shape(test_data), np.shape(test_labels))
    print("======Load finished======")

    print("======Shuffling data======")
    indices = np.random.permutation(len(train_data))
    train_data = (train_data[indices])
    train_labels = train_labels[indices]
    indices = np.random.permutation(len(test_data))
    test_data = (test_data[indices])
    test_labels = test_labels[indices]


    print("======Prepare Finished======")

    return train_data, train_labels, test_data, test_labels

def normalization(data):
    _range = np.max(data) - np.min(data)
    out = (data - np.min(data)) / _range

    return out

def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    out = 0.5 * ((data - mu) / sigma) + 0.5

    # print(np.mean(out, axis=0), np.std(out, axis=0))
    return out

def sigmoid(X):
	return 1.0 / (1 + np.exp(-(X)))
