# date generator


import numpy as np
import pickle
import os
import math
import cv2
from keras.utils import to_categorical


class cifar10_gen:
    def __init__(self, path, batch_size=128):
        self.path = path
        self.batch_size = batch_size

    def load_file(self, filename):
        with open(filename, 'rb') as file:
            data = pickle.load(file, encoding='latin1')
        return data


    def train_generator(self):
        data = self.load_file(os.path.join(self.path, 'train'))
        images = data['data']
        labels = data['labels']
        print()
        n_iter = math.ceil(len(labels) / self.batch_size)
        num = 0
        while True:
            print(images.shape)
            for i in range(n_iter):
                image = images[i * self.batch_size: (i + 1) * self.batch_size][:]
                label = labels[i * self.batch_size: (i + 1) * self.batch_size][:]

                num += len(label)
                print(len(label))
                print(num)

                image = np.array(image).reshape((len(label), 32, 32, 3))

                mean = np.mean(image, axis=3)
                for i in range(3):
                    image[:, :, :, i] = image[:, :, :, i] - mean

                image_ref = image[:, :, :, ::-1]

                new_image = np.zeros((len(label), 224, 224, 3))
                new_image_ref = np.zeros(new_image.shape)

                print(new_image.shape)

                for j in range(len(label)):
                    new_image[j] = cv2.resize(image[j], (224, 224))

                label = to_categorical(label, 10)
                label_ref = label

                yield new_image, label

                for j in range(len(label)):
                    new_image_ref[j] = cv2.resize(image_ref[j], (224, 224))

                yield new_image_ref, label_ref

    def valid(self):
        data = self.load_file(os.path.join(self.path, 'test'))
        labels = data['labels']
        images = np.array(data['data']).reshape((len(labels), 32, 32, 3))
        new_images = np.zeros((len(labels), 224, 224, 3))
        for j in range(self.batch_size):
            new_images[j] = cv2.resize(images[j], (224, 224))
        labels = to_categorical(labels, 10)
        return new_images, labels