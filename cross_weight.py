import argparse
import glob
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--process', help='Process', default='train')
parser.add_argument('-data', '--dataset', help='Dataset', default='ucf101')
parser.add_argument('-b', '--batch', help='Batch size', default=16, type=int)
parser.add_argument('-c', '--classes', help='Number of classes', default=101, type=int)
parser.add_argument('-e', '--epoch', help='Number of epochs', default=5, type=int)
parser.add_argument('-dropout', '--dropout', help='Dropout', default=0.8, type=float)
parser.add_argument('-r', '--retrain', help='Number of old epochs when retrain', default=0, type=int)
parser.add_argument('-cross', '--cross', help='Cross fold', default=1, type=int)
parser.add_argument('-s', '--summary', help='Show model', default=0, type=int)
parser.add_argument('-lr', '--lr', help='Learning rate', default=1e-3, type=float)
parser.add_argument('-decay', '--decay', help='Decay', default=0.0, type=float)
parser.add_argument('-fine', '--fine', help='Fine-tuning', default=1, type=int)
parser.add_argument('-n', '--neural', help='LSTM neural', default=256, type=int)
parser.add_argument('-t', '--temporal', help='Temporal rate', default=1, type=int)

args = parser.parse_args()
print (args)

import numpy as np
import sys
import config
import models
import keras as K
from keras import optimizers

process = args.process
old_epochs = 0
batch_size = args.batch
classes = args.classes
epochs = args.epoch
cross_index = args.cross
dataset = args.dataset
temp_rate = args.temporal

seq_len = 3
n_neurons = args.neural
dropout = args.dropout
# pre_file = 'inception_spatial2fc_{}'.format(n_neurons)
pre_file = 'incept_temporal{}_{}'.format(temp_rate,n_neurons)

spatial_a = models.InceptionSpatial_a(
                    n_neurons=n_neurons, seq_len=seq_len, classes=classes, 
                    weights=None, dropout=dropout, fine=1, retrain=False,
                    pre_file=pre_file,old_epochs=old_epochs,cross_index=cross_index)
spatial_b = models.InceptionSpatial_b(n_neurons=n_neurons, seq_len=seq_len, classes=classes, 
                    weights=None, dropout=dropout, fine=1, retrain=False,
                    pre_file=pre_file,old_epochs=old_epochs,cross_index=cross_index,
                    input_shape=spatial_a.output_shape, input_tensor=spatial_a.output)

spatial = models.InceptionSpatial(
                    n_neurons=n_neurons, seq_len=seq_len, classes=classes, 
                    weights=None, dropout=dropout, fine=False, retrain=False,
                    pre_file=pre_file,old_epochs=old_epochs,cross_index=cross_index)

spatial_a.summary()
spatial_b.summary()
spatial.summary()

# print(glob.glob('weights/inception_spatial2fc_{}_{}e_cr{}.h5'.format(n_neurons,pre_train[0],cross_index)[-1]))    
spatial.load_weights('weights/sinception_spatial2fc_256-62-0.8554.hdf5')
print("Load")

spatial_a.layers[0].set_weights(spatial.layers[0].get_weights())

spatial_a.layers[1].set_weights(spatial.layers[1].get_weights())

spatial_a.layers[2].set_weights(spatial.layers[2].get_weights())

spatial_a.layers[3].set_weights(spatial.layers[5].get_weights())

#spatial_a.save_weights('./weights/a_sincept_temporal1_256-55-0.8300.hdf5')
print("a")

spatial_b.layers[2].set_weights(spatial.layers[8].get_weights())

spatial_b.layers[3].set_weights(spatial.layers[11].get_weights())

spatial_b.layers[4].set_weights(spatial.layers[14].get_weights())

spatial_b.layers[5].set_weights(spatial.layers[17].get_weights())

spatial_b.layers[6].set_weights(spatial.layers[20].get_weights())

spatial_b.save_weights('./weights/b_sinception_spatial2fc_256-62-0.8554.hdf5')

# model_t =  models.InceptionTemporal(
#                     n_neurons=n_neurons, seq_len=seq_len, classes=classes, 
#                     weights=None, dropout=dropout, fine=False, retrain=False,
#                     pre_file=pre_file,old_epochs=old_epochs,cross_index=cross_index)

# model_s.summary()

# print(glob.glob('weights/weights-improvement-{:2d}-*.hdf5'.format(epochs))[-1])    
# model_s.load_weights(glob.glob('weights/weights-improvement-{:2d}-*.hdf5'.format(epochs))[-1])

# weights = model_s.layers[0].get_weights()
# weights_0 = weights[0]
# print(np.asarray(weights_0).shape)
# weights_0 = np.mean(weights_0, axis=2).reshape(3,3,1,32)
# weights_0 = np.repeat(weights_0, 20, axis=2)
# print(weights_0.shape)

# weights[0] = weights_0
# for i in range(len(model_t.layers)):
#    print(model_t.layers[i].name)
#    if i in [0,10,13,16,19]:
#       model_t.layers[i].set_weights(weights)
#    else:
#       model_t.layers[i].set_weights(model_s.layers[i].get_weights())
# #K.set_value(model_t.layers[0].weights[0], weights_0)
# model_t.save_weights('./weights/iincept_temporal1_256_1e_cr1.h5')
