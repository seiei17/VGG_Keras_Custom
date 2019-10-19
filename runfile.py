# this is run file.


from VGG_model import vgg_16

model = vgg_16((224, 224, 3,), 10, '../database/cifar10_new/',  epochs=50)

model.train()
