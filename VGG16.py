# this is the vgg16 model file.

import keras
from keras.layers import Conv2D as Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Dropout
from keras.layers import BatchNormalization

from keras.models import Model

from keras.regularizers import l2
from keras.initializers import TruncatedNormal as tn
from keras.initializers import Zeros as zero
from keras.initializers import Constant as cons


def vgg16(input_shape=(224, 224, 3,), num_classes=10, w_decay=0.0005):
    '''
    this is vgg16 model, which has 16 weight layers
    :param input_shape: is (224, 224, 3,)
    :param num_classes: using cifar 10 dataset, num_classes=10
    :param w_decay: is the l2 regularizers
    :return: model of vgg16
    '''
    net = {}
    input_tensor = Input(input_shape)
    net['input'] = input_tensor

    # layer 1
    net['conv1'] = Convolution2D(64, (3, 3),
                                 padding='same',
                                 activation='relu',
                                 kernel_regularizer=l2(w_decay),
                                 kernel_initializer=tn(0, 0.1),
                                 bias_initializer=zero(),
                                 name='conv1')(net['input'])

    # layer 2
    net['conv2'] = Convolution2D(64, (3, 3),
                                 padding='same',
                                 activation='relu',
                                 kernel_regularizer=l2(w_decay),
                                 kernel_initializer=tn(0, 0.1),
                                 bias_initializer=zero(),
                                 name='conv2')(net['conv1'])
    net['pool2'] = MaxPool2D((2, 2), strides=2,
                             name='pool2')(net['conv2'])

    # layer 3
    net['conv3'] = Convolution2D(128, (3, 3),
                                 padding='same',
                                 activation='relu',
                                 kernel_regularizer=l2(w_decay),
                                 kernel_initializer=tn(0, 0.1),
                                 bias_initializer=zero(),
                                 name='conv3')(net['pool2'])

    # layer 4
    net['conv4'] = Convolution2D(128, (3, 3),
                                 padding='same',
                                 activation='relu',
                                 kernel_regularizer=l2(w_decay),
                                 kernel_initializer=tn(0, 0.1),
                                 bias_initializer=zero(),
                                 name='conv4')(net['conv3'])
    net['pool4'] = MaxPool2D((2, 2), strides=2,
                             name='pool4')(net['conv4'])

    # layer 5
    net['conv5'] = Convolution2D(256, (3, 3),
                                 padding='same',
                                 activation='relu',
                                 kernel_regularizer=l2(w_decay),
                                 kernel_initializer=tn(0, 0.1),
                                 bias_initializer=zero(),
                                 name='conv5')(net['pool4'])

    # layer 6
    net['conv6'] = Convolution2D(256, (3, 3),
                                 padding='same',
                                 activation='relu',
                                 kernel_regularizer=l2(w_decay),
                                 kernel_initializer=tn(0, 0.1),
                                 bias_initializer=zero(),
                                 name='conv6')(net['conv5'])

    # layer 7
    net['conv7'] = Convolution2D(256, (3, 3),
                                 padding='same',
                                 activation='relu',
                                 kernel_regularizer=l2(w_decay),
                                 kernel_initializer=tn(0, 0.1),
                                 bias_initializer=zero(),
                                 name='conv7')(net['conv6'])
    net['pool7'] = MaxPool2D((2, 2), strides=2,
                             name='pool7')(net['conv7'])

    # layer 8
    net['conv8'] = Convolution2D(512, (3, 3),
                                 padding='same',
                                 activation='relu',
                                 kernel_regularizer=l2(w_decay),
                                 kernel_initializer=tn(0, 0.1),
                                 bias_initializer=zero(),
                                 name='conv8')(net['pool7'])

    # layer 9
    net['conv9'] = Convolution2D(512, (3, 3),
                                 padding='same',
                                 activation='relu',
                                 kernel_regularizer=l2(w_decay),
                                 kernel_initializer=tn(0, 0.1),
                                 bias_initializer=zero(),
                                 name='conv9')(net['conv8'])

    # layer 10
    net['conv10'] = Convolution2D(512, (3, 3),
                                  padding='same',
                                  activation='relu',
                                  kernel_regularizer=l2(w_decay),
                                  kernel_initializer=tn(0, 0.1),
                                  bias_initializer=zero(),
                                  name='conv10')(net['conv9'])
    net['pool10'] = MaxPool2D((2, 2), strides=2,
                              name='pool10')(net['conv10'])

    # layer 11
    net['conv11'] = Convolution2D(512, (3, 3),
                                  padding='same',
                                  activation='relu',
                                  kernel_regularizer=l2(w_decay),
                                  kernel_initializer=tn(0, 0.1),
                                  bias_initializer=zero(),
                                  name='conv11')(net['pool10'])

    # layer 12
    net['conv12'] = Convolution2D(512, (3, 3),
                                  padding='same',
                                  activation='relu',
                                  kernel_regularizer=l2(w_decay),
                                  kernel_initializer=tn(0, 0.1),
                                  bias_initializer=zero(),
                                  name='conv12')(net['conv11'])

    # layer 13
    net['conv13'] = Convolution2D(512, (3, 3),
                                  padding='same',
                                  activation='relu',
                                  kernel_regularizer=l2(w_decay),
                                  kernel_initializer=tn(0, 0.1),
                                  bias_initializer=zero(),
                                  name='conv13')(net['conv12'])
    net['pool13'] = MaxPool2D((2, 2), strides=2,
                              name='pool13')(net['conv13'])

    # layer 14
    net['flat14'] = Flatten(name='flat14')(net['pool13'])
    net['fc14'] = Dense(4096,
                        activation='relu',
                        kernel_regularizer=l2(w_decay),
                        kernel_initializer=tn(0, 0.001),
                        name='fc14')(net['flat14'])
    net['bn14'] = BatchNormalization(name='bn14')(net['fc14'])
    net['dp14'] = Dropout(0.4, name='dp14')(net['bn14'])

    # layer 15
    net['fc15'] = Dense(4096,
                        activation='relu',
                        kernel_regularizer=l2(w_decay),
                        kernel_initializer=tn(0, 0.001),
                        name='fc15')(net['dp14'])
    net['bn15'] = BatchNormalization(name='bn15')(net['fc15'])
    net['dp15'] = Dropout(0.4, name='dp15')(net['bn15'])

    # layer 16
    net['output'] = Dense(num_classes,
                          activation='softmax',
                          kernel_initializer=tn(0, 0.001),
                          name='output')(net['dp15'])

    return Model(net['input'], net['output'])
