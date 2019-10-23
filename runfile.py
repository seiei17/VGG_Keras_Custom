# this is the runfile

import keras
import keras.backend as K
import tensorflow
import math

from VGG16 import vgg16
from data_generator import cifar10_gen

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


# now using cifar 10
path = '../database/cifar10/'
num_classes = 10
batch_size = 256
steps = math.ceil(10000 / batch_size)
epochs = 30

model = vgg16((224, 224, 3,), num_classes)
gen = cifar10_gen(path, batch_size)
opt = keras.optimizers.SGD(0.01, 0.9)
callback = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                             factor=0.1,
                                             patience=5,
                                             verbose=1)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(generator=gen.train_generator(),
                    steps_per_epoch=5 * steps,
                    epochs=epochs,
                    verbose=1,
                    callbacks=[callback],
                    validation_data=gen.valid_generator(),
                    validation_steps=steps)

print()
print('now saving the check point.')
keras.models.save_model(model, './checkpoint/vgg16_checkpoint_{}.h5'.format(epochs))