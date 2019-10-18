# this is run file.


from VGG_model import vgg_16

model = vgg_16((224, 224, 3,), 10, '../database/cifar10_mix/',
               epochs=50,
               batch_size=256)

model.train()