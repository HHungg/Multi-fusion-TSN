import numpy as np
import sys
import config
import models
import keras as K
from keras import optimizers
from keras.applications import InceptionV3
from inceptionv3 import *

inception = TempInceptionV3(input_shape=(299,299,20), pooling='avg', include_top=False, weights='imagenet',)

inceptionv3a = Inception_temp_v3a(input_shape=(299,299,20), weights=None)
inceptionv3b = Inception_v3b(input_shape=inceptionv3a.output_shape, input_tensor=inceptionv3a.output, weights=None)
inceptionv3c = Inception_v3c(input_shape=inceptionv3b.output_shape, input_tensor=inceptionv3b.output, weights=None, pooling=None)

#print(inception.summary())
print(inceptionv3a.summary())
#print(inceptionv3c.summary())

for i in range(len(inceptionv3a.layers)):
    inceptionv3a.layers[i].set_weights(inception.layers[i+1].get_weights())
inceptionv3a.save_weights('./data/Temp_InceptionV3a.h5')
print('v3a')

for i in range(len(inceptionv3b.layers)):
    if i!=0:
        inceptionv3b.layers[i].set_weights(inception.layers[i+len(inceptionv3a.layers)-1].get_weights())
# inceptionv3b.save_weights('./data/InceptionV3b.h5')
print('v3b')

for i in range(len(inceptionv3c.layers)):
    if i!=0:
        inceptionv3c.layers[i].set_weights(inception.layers[i+len(inceptionv3a.layers)+len(inceptionv3b.layers)-2].get_weights())
# inceptionv3c.save_weights('./data/InceptionV3c3d.h5')
