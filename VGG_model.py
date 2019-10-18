# the is model file.

import keras
import tensorflow
import keras.backend as K

from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Dropout
from keras.regularizers import l2
from keras.models import Model
from keras.models import save_model

from keras.callbacks import ReduceLROnPlateau
# from keras.preprocessing.image import ImageDataGenerator
import math
import os

from data_generator import cifar10_gen

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

class vgg_model():

    def __init__(self,
                 input_shape,
                 num_classes,
                 data_path,
                 epochs=50,
                 batch_size=128,
                 weight_decay=0.0005):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.path = data_path
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs

    def conv1(self, layers, name, input, activation='relu'):
        return Convolution2D(layers, (1, 1),
                             strides=1,
                             activation=activation,
                             kernel_regularizer=l2(self.weight_decay),
                             name=name)(input)

    def conv3(self, layers, name, input, activation='relu'):
        return Convolution2D(layers, (3, 3),
                             strides=1,
                             padding='same',
                             activation=activation,
                             kernel_regularizer=l2(self.weight_decay),
                             name=name)(input)

    def maxpool(self, name, input):
        return MaxPool2D((2, 2),
                         strides=2,
                         name=name)(input)

    def dense(self, unit, name, input, activation='relu'):
        return Dense(unit, activation=activation, name=name)(input)

    def vgg(self):
        net = {}
        input_tensor = Input(self.input_shape)
        net['input'] = input_tensor
        net['output'] = input_tensor
        return Model(net['input'], net['output'])

    def train(self):
        gen = cifar10_gen('../database/cifar10_mix/', batch_size=self.batch_size)
        optimizer = keras.optimizers.sgd(0.01, 0.9)
        callback = ReduceLROnPlateau(monitor='val_loss', factor=0.1, verbose=1)
        model = self.vgg()
        x_val, y_val = gen.valid()
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit_generator(gen.train_generator(),
                            steps_per_epoch=2*2*math.ceil(25000/self.batch_size),
                            epochs=self.epochs,
                            verbose=1,
                            callbacks=[callback],
                            validation_data=(x_val, y_val),
                            validation_steps=self.batch_size)
        save_model(model, './vgg16_{}.h5'.format(self.epochs))


class vgg_16(vgg_model):

    def vgg(self):
        net = {}
        input_tensor = Input(self.input_shape)
        net['input'] = input_tensor

        # layer 1
        net['conv1'] = self.conv3(64, 'conv1', net['input'])

        # layer 2
        net['pool2'] = self.maxpool('pool2', net['conv1'])

        # layer 3
        net['conv3'] = self.conv3(128, 'conv3', net['pool2'])

        # layer 4
        net['pool4'] = self.maxpool('pool4', net['conv3'])

        # layer 5
        net['conv5'] = self.conv3(256, 'conv5', net['pool4'])

        # layer 6
        net['conv6'] = self.conv3(256, 'conv6', net['conv5'])

        # layer 7
        net['pool7'] = self.maxpool('pool7', net['conv6'])

        # layer 8
        net['conv8'] = self.conv3(512, 'conv8', net['pool7'])

        # layer 9
        net['conv9'] = self.conv3(512, 'conv9', net['conv8'])

        # layer 10
        net['pool10'] = self.maxpool('pool10', net['conv9'])

        # layer 11
        net['conv11'] = self.conv3(512, 'conv11', net['pool10'])

        # layer 12
        net['conv12'] = self.conv3(512, 'conv12', net['conv11'])

        # layer 13
        net['pool13'] = self.maxpool('pool13', net['conv12'])
        net['flat13'] = Flatten(name='flat13')(net['pool13'])

        # layer 14
        net['fc14'] = self.dense(4096, 'fc14', net['flat13'])
        net['drop14'] = Dropout(0.5, name='drop14')(net['fc14'])

        # layer 15
        net['fc15'] = self.dense(4096, 'fc15', net['drop14'])
        net['drop15'] = Dropout(0.5, name='drop15')(net['fc15'])

        # layer 16
        net['fc16'] = self.dense(self.num_classes, 'fc16',
                                 activation='softmax',
                                 input=net['drop15'])

        return Model(net['input'], net['fc16'])
