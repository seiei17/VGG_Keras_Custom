# make 5 cifar files to 1 file


import pickle
import os
import numpy as np

read_prefix = '../database/cifar10/'


def load_data(filepath):
    with open(filepath, 'rb') as file:
        return pickle.load(file, encoding='latin1')


def dump_data(filepath, writing_data):
    with open(filepath, 'wb') as file:
        pickle.dump(writing_data, file)

# mix train data


images = []
labels = []
for i in range(5):
    path = os.path.join(read_prefix, 'data_batch_{}'.format(i+1))
    data = load_data(path)
    image = data['data']
    label = data['labels']
    images.append(image)
    labels.append(label)

images = np.array(images).reshape(50000, 3072)
labels = np.array(labels).reshape(50000, )

zipdata = dict(zip(['data', 'labels'], [images, labels]))

dump_data('../database/cifar10_mix/train', zipdata)