# to divide cifar_10 into liittle package

import pickle
import os


def load_file(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file, encoding='latin1')
    return data


def write_file(filename, data):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

# for cifar_10 data, every batch has 10,000 pics
# divide every batch into 20 new batches
# every new batch has 5 pics


def cifar10_divide():

    file_read_prefix = '../database/cifar10/'
    file_write_prefix = '../database/cifar10_new/'

    # divide train data
    num = [1, 2, 3, 4, 5]
    k = 0
    size = 500
    for i in num:

        read_path = os.path.join(file_read_prefix, 'data_batch_{}'.format(i))
        data = load_file(read_path)
        images = data['data']
        labels = data['labels']
        for j in range(20):
            write_data = dict(zip(['images', 'labels'], [images[j * size: (j + 1)* size][:],
                                                          labels[j * size: (j + 1)* size][:]]))
            write_path = os.path.join(file_write_prefix, 'data_batch_{}'.format(k))
            k += 1
            write_file(write_path, write_data)

    # divede test data
    k = 0
    read_path = os.path.join(file_read_prefix, 'test_batch')
    data = load_file(read_path)
    images = data['data']
    labels = data['labels']
    for i in range(20):
        write_data = dict(zip(['images', 'labels'], [images[i * size: (i + 1)* size][:],
                                                     labels[i * size: (i + 1)* size][:]]))
        write_path = os.path.join(file_write_prefix, 'test_batch_{}'.format(k))
        k += 1
        write_file(write_path, write_data)


def cifar100_divide():
    file_read_prefix = '../datasets/cifar100/'
    file_write_prefix = '../datasets/new_cifar100/'

    # divide train data
    k = 0

    read_path = os.path.join(file_read_prefix, 'train')
    data = load_file(read_path)
    images = data['data']
    labels = data['fine_labels']
    for j in range(100):
        write_data = dict(zip(['images', 'labels'], [images[j * 500: (j + 1) * 500][:],
                                                         labels[j * 500: (j + 1) * 500][:]]))
        write_path = os.path.join(file_write_prefix, 'data_batch_{}'.format(k))
        k += 1
        write_file(write_path, write_data)

    # divede test data
    k = 0
    read_path = os.path.join(file_read_prefix, 'test')
    data = load_file(read_path)
    images = data['data']
    labels = data['fine_labels']
    for i in range(20):
        write_data = dict(zip(['images', 'labels'], [images[i * 500: (i + 1) * 500][:],
                                                     labels[i * 500: (i + 1) * 500][:]]))
        write_path = os.path.join(file_write_prefix, 'test_batch_{}'.format(k))
        k += 1
        write_file(write_path, write_data)




cifar10_divide()
# cifar100_divide()
