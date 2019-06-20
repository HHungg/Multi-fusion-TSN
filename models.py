import config
import glob
import pickle
import random
import time
import math
import numpy as np
import get_data as gd
import keras.backend as K
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, ZeroPadding2D, Concatenate, BatchNormalization, add
from keras.layers import TimeDistributed, Activation, AveragePooling1D, GlobalAveragePooling1D, GlobalAveragePooling2D, GlobalAveragePooling3D, AveragePooling2D
from keras.layers import LSTM, GlobalAveragePooling1D, Reshape, MaxPooling1D, Conv2D, Conv3D, MaxPooling3D
from keras.layers import Input, Lambda, Average, average
from keras.applications.mobilenet import MobileNet
from keras.applications.inception_v3 import InceptionV3
from keras.applications.densenet import DenseNet169, DenseNet121, DenseNet201
from keras.applications.resnet50 import ResNet50
from inceptionv3 import TempInceptionV3, Inception_v3a, Inception_v3b, Inception_v3c, conv2d_bn, Inception_temp_v3a
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
from keras import optimizers
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, CSVLogger

server = config.server()
data_output_path = config.data_output_path()

def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.5
    epochs_drop = 5.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

def relu6(x):
    return K.relu(x, max_value=6)

def concat_fusion(x):
    a = x[0]
    b = x[1]

    c = K.concatenate([a[:, :, :, :, :1], b[:, :, :, :, :1]], axis=4)
    for i in range(a.shape[-1]+1):
        if(i > 1):
            c = K.concatenate([c[:, :, :, :, :], a[:, :, :, :, i-1:i], b[:, :, :, :, i-1:i]], axis=4)

    return c

def concat_fusion_output_shape(input_shape):
    return (input_shape[0][0], 3, 8, 8, 4096)

def InceptionSpatial_a(n_neurons=256, seq_len=3, classes=101, weights='imagenet', 
    dropout=0.5, fine=True, retrain=False, pre_file='',old_epochs=0,cross_index=1):

    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    # inception = InceptionV3(input_shape=(299,299,3), pooling='avg', include_top=False, weights=weights)
    inceptionv3a = Inception_v3a(input_shape=(299,299,3), weights=weights)
    inceptionv3b = Inception_v3b(input_shape=inceptionv3a.output_shape, input_tensor=inceptionv3a.output, weights=weights)
    inceptionv3c = Inception_v3c(input_shape=inceptionv3b.output_shape, input_tensor=inceptionv3b.output, weights=weights
    #, pooling=None
    )

    input = Input(shape=(seq_len, 299,299,3))
    model = TimeDistributed(inceptionv3a)(input)
    model = TimeDistributed(inceptionv3b)(model)
    model = TimeDistributed(inceptionv3c)(model)

    result_model = Model(inputs=input, outputs=model, name="InceptionSpatial_a")

    return result_model

def InceptionSpatial_b(n_neurons=256, seq_len=3, classes=101, weights='imagenet', 
    dropout=0.5, fine=True, retrain=False, pre_file='',old_epochs=0,cross_index=1, 
    input_tensor=None, input_shape=None, depth=2048):

    if input_shape is None:
        input_shape = (None,None,None,depth)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3

    model = Input(tensor=img_input)
    # model = TimeDistributed(GlobalAveragePooling2D())(model)
    model = LSTM(n_neurons, return_sequences=True)(model)
    model = Flatten()(model)
    model = Dense(256, activation='relu')(model)
    model = Dropout(dropout)(model)
    model = Dense(classes, activation='softmax')(model)
    inputs = img_input

    result_model = Model(inputs=inputs, outputs=model,
                         name="InceptionSpatial_b")

    return result_model

def InceptionSpatial(n_neurons=512, seq_len=3, classes=101, weights='imagenet', 
    dropout=0.5, fine=True, retrain=False, pre_file='',old_epochs=0,cross_index=1):

    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    # inception = InceptionV3(input_shape=(299,299,3), pooling='avg', include_top=False, weights=weights)
    inceptionv3a = Inception_v3a(input_shape=(299,299,3), weights=weights)
    inceptionv3b = Inception_v3b(input_shape=inceptionv3a.output_shape, input_tensor=inceptionv3a.output, weights=weights)
    inceptionv3c = Inception_v3c(input_shape=inceptionv3b.output_shape, input_tensor=inceptionv3b.output, weights=weights, pooling=None)

    input = Input(shape=(seq_len, 299,299,3))
    model = TimeDistributed(inceptionv3a)(input)
    loss1 = TimeDistributed(AveragePooling2D((5, 5), strides=(3, 3), padding='same'))(model)
    loss1 = TimeDistributed(Conv2D(128, (1,1), strides=(1, 1), padding='same', use_bias=False))(loss1)
    loss1 = TimeDistributed(BatchNormalization(axis=bn_axis, scale=False))(loss1)
    loss1 = TimeDistributed(Activation('relu'))(loss1)
    loss1 = GlobalAveragePooling3D()(loss1)
    loss1 = Dense(classes, activation='softmax')(loss1)

    model = TimeDistributed(inceptionv3b)(model)
    loss2 = TimeDistributed(AveragePooling2D((5, 5), strides=(3, 3), padding='same'))(model)
    loss2 = TimeDistributed(Conv2D(128, (1,1), strides=(1, 1), padding='same', use_bias=False))(loss2)
    loss2 = TimeDistributed(BatchNormalization(axis=bn_axis, scale=False))(loss2)
    loss2 = TimeDistributed(Activation('relu'))(loss2)
    loss2 = GlobalAveragePooling3D()(loss2)
    loss2 = Dense(classes, activation='softmax')(loss2)

    model = TimeDistributed(inceptionv3c)(model)
    # model = LSTM(n_neurons, return_sequences=True)(model)
    # model = GlobalAveragePooling1D()(model)
    # model = Flatten()(model)
    model = Conv3D(256, (3, 3, 3), activation='relu', padding='same', name='conv3')(model)
    model = MaxPooling3D(pool_size=(3, 2, 2), strides=(1, 2, 2), padding='valid', name='pool3')(model)
    model = Flatten()(model)
    model = Dense(256, activation='relu')(model)
    model = Dropout(dropout)(model)
    model = Dense(classes, activation='softmax')(model)

    result_model = Model(inputs=input,
			 outputs=[
                 #loss1, loss2,
                  model],
                         name="InceptionSpatial")

    if retrain:
        print(glob.glob('weights/' + pre_file + '-{:02d}-*.hdf5'.format(old_epochs))[-1])
        result_model.load_weights(glob.glob('weights/' + pre_file + '-{:02d}-*.hdf5'.format(old_epochs))[-1])

    return result_model

def InceptionTemporal_a(n_neurons=256, seq_len=3, classes=101, weights='imagenet', 
    dropout=0.5, fine=True, retrain=False, pre_file='',old_epochs=0,cross_index=1):

    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    # inception = InceptionV3(input_shape=(299,299,3), pooling='avg', include_top=False, weights=weights)
    inceptionv3a = Inception_temp_v3a(input_shape=(299,299,20), weights=weights)
    inceptionv3b = Inception_v3b(input_shape=inceptionv3a.output_shape, input_tensor=inceptionv3a.output, weights=weights)
    inceptionv3c = Inception_v3c(input_shape=inceptionv3b.output_shape, input_tensor=inceptionv3b.output, weights=weights
    #, pooling=None
    )

    input = Input(shape=(seq_len, 299,299,20))
    model = TimeDistributed(inceptionv3a)(input)
    model = TimeDistributed(inceptionv3b)(model)
    model = TimeDistributed(inceptionv3c)(model)

    result_model = Model(inputs=input, outputs=model, name="InceptionTemporal_a")

    return result_model

def InceptionTemporal_b(n_neurons=256, seq_len=3, classes=101, weights='imagenet', 
    dropout=0.5, fine=True, retrain=False, pre_file='',old_epochs=0,cross_index=1, 
    input_tensor=None, input_shape=None, depth=2048):

    if input_shape is None:
        input_shape = (None,None,depth)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3

    model = Input(tensor=img_input)
    # model = TimeDistributed(GlobalAveragePooling2D())(model)
    model = LSTM(n_neurons, return_sequences=True)(model)
    model = Flatten()(model)
    model = Dense(256, activation='relu')(model)
    model = Dropout(dropout)(model)
    model = Dense(classes, activation='softmax')(model)
    inputs = img_input 

    result_model = Model(inputs=inputs, outputs=model,
                         name="InceptionTemporal_b")

    return result_model

def InceptionTemporal(n_neurons=256, seq_len=3, classes=101, weights='imagenet', 
    dropout=0.5, fine=True, retrain=False, pre_file='',old_epochs=0,cross_index=1):
    
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    # inception = InceptionV3(input_shape=(299,299,3), pooling='avg', include_top=False, weights=weights)
    inceptionv3a = Inception_temp_v3a(input_shape=(299,299,20), weights=weights)
    inceptionv3b = Inception_v3b(input_shape=inceptionv3a.output_shape, input_tensor=inceptionv3a.output, weights=weights)
    inceptionv3c = Inception_v3c(input_shape=inceptionv3b.output_shape, input_tensor=inceptionv3b.output, weights=weights)

    input = Input(shape=(seq_len, 299,299,20))
    model = TimeDistributed(inceptionv3a)(input)
    loss1 = TimeDistributed(AveragePooling2D((5, 5), strides=(3, 3), padding='same'))(model)
    loss1 = TimeDistributed(Conv2D(128, (1,1), strides=(1, 1), padding='same', use_bias=False))(loss1)
    loss1 = TimeDistributed(BatchNormalization(axis=bn_axis, scale=False))(loss1)
    loss1 = TimeDistributed(Activation('relu'))(loss1)
    loss1 = GlobalAveragePooling3D()(loss1)
    loss1 = Dense(classes, activation='softmax')(loss1)

    model = TimeDistributed(inceptionv3b)(model)
    loss2 = TimeDistributed(AveragePooling2D((5, 5), strides=(3, 3), padding='same'))(model)
    loss2 = TimeDistributed(Conv2D(128, (1,1), strides=(1, 1), padding='same', use_bias=False))(loss2)
    loss2 = TimeDistributed(BatchNormalization(axis=bn_axis, scale=False))(loss2)
    loss2 = TimeDistributed(Activation('relu'))(loss2)
    loss2 = GlobalAveragePooling3D()(loss2)
    loss2 = Dense(classes, activation='softmax')(loss2)

    model = TimeDistributed(inceptionv3c)(model)
    model = LSTM(n_neurons, return_sequences=True)(model)
    model = Flatten()(model)
    model = Dense(256, activation='relu')(model)
    model = Dropout(dropout)(model)
    model = Dense(classes, activation='softmax')(model)

    result_model = Model(inputs=input,
                         outputs=[loss1, loss2, model],
                         name="InceptionSpatial")

    if retrain:
        print(glob.glob('weights/' + pre_file + '-{:02d}*.hdf5'.format(old_epochs))[-1])
        result_model.load_weights(glob.glob('weights/' + pre_file + '-{:02d}*.hdf5'.format(old_epochs))[-1])

    return result_model

def InceptionMultistream1(n_neurons=256, seq_len=3, classes=101, weights='imagenet', 
    dropout=0.8, fine=True, retrain=False, pre_file='',old_epochs=0,cross_index=1, pre_train=None,temp_rate=1):

    if weights != 'imagenet':
        weight = None
    else:
        weight = weights

    spatial_a = InceptionSpatial_a(
                    n_neurons=n_neurons, seq_len=seq_len, classes=classes, 
                    weights=weight, dropout=dropout, fine=fine, retrain=False,
                    pre_file=pre_file,old_epochs=old_epochs,cross_index=cross_index)
    spatial_b = InceptionSpatial_b(n_neurons=n_neurons, seq_len=seq_len, classes=classes, 
                     weights=weight, dropout=dropout, fine=fine, retrain=False,
                     pre_file=pre_file,old_epochs=old_epochs,cross_index=cross_index,
                     input_shape=spatial_a.output_shape, input_tensor=spatial_a.output)

    # if (weights == 'pretrain') & (not retrain):
    #     print(glob.glob('weights/' + 'a_sinception_spatial2fc_256' + '-{:02d}-*.hdf5'.format(pre_train[0]))[-1])
    #     spatial_a.load_weights(glob.glob('weights/' + 'a_sinception_spatial2fc_256' + '-{:02d}-*.hdf5'.format(pre_train[0]))[-1])
    #     spatial_b.load_weights(glob.glob('weights/' + 'b_sinception_spatial2fc_256' + '-{:02d}-*.hdf5'.format(pre_train[0]))[-1])
    #     print ('load spatial weights')

    for layer in spatial_a.layers:
        layer.trainable = False

    temporal_a = InceptionTemporal_a(
                    n_neurons=n_neurons, seq_len=seq_len, classes=classes, 
                    weights=weight, dropout=dropout, fine=fine, retrain=False,
                    pre_file=pre_file,old_epochs=old_epochs,cross_index=cross_index)
    temporal_b = InceptionTemporal_b(
                    n_neurons=n_neurons, seq_len=seq_len, classes=classes, 
                    weights=weight, dropout=dropout, fine=fine, retrain=False,
                    pre_file=pre_file,old_epochs=old_epochs,cross_index=cross_index,
                    input_shape=temporal_a.output_shape, input_tensor=temporal_a.output)
    
    # if (weights == 'pretrain') & (not retrain):
    #     print(glob.glob('weights/' + 'a_sincept_temporal1_256' + '-{:02d}-*.hdf5'.format(pre_train[1]))[-1])
    #     temporal_a.load_weights(glob.glob('weights/' + 'a_sincept_temporal1_256' + '-{:02d}-*.hdf5'.format(pre_train[1]))[-1])
    #     temporal_b.load_weights(glob.glob('weights/' + 'b_sincept_temporal1_256' + '-{:02d}-*.hdf5'.format(pre_train[1]))[-1])
    #     print ('load temporal weights')

    for layer in temporal_a.layers:
        layer.trainable = False

    input1 = Input(shape=(seq_len, 299,299,3))
    input2 = Input(shape=(seq_len, 299,299,20))
    spatial = spatial_a(input1)
    temporal = temporal_a(input2)
    model_spatial_a = Model(inputs=input1, outputs=spatial)
    model_temporal_a = Model(inputs=input2, outputs=temporal)

    model_spatial_b = Model(inputs=input1, outputs=spatial_b(model_spatial_a.output))
    model_temporal_b = Model(inputs=input2, outputs=temporal_b(model_temporal_a.output))

    concat = Concatenate()([model_spatial_b.output, model_temporal_b.output])
    concat = Flatten()(concat)
    concat = Dense(256, activation='relu')(concat)
    concat = BatchNormalization()(concat)
    concat = Dropout(dropout)(concat)
    concat = Dense(classes, activation='softmax')(concat)
    result_model = Model(inputs=[input1, input2], outputs=concat)
    print(result_model.output_shape)

    if retrain:
        print(glob.glob('weights/' + pre_file + '-{:02d}-*.hdf5'.format(old_epochs))[-1])
        result_model.load_weights(glob.glob('weights/' + pre_file + '-{:02d}-*.hdf5'.format(old_epochs))[-1])

    return result_model


def foo(x):
    a = x[0]
    b = x[1]
    return (a + b)/2

def foo_output_shape(input_shape):
    return (input_shape[0][0], 101)

def InceptionMultistream2(n_neurons=256, seq_len=3, classes=101, weights='imagenet', 
    dropout=0.8, fine=True, retrain=False, pre_file='',old_epochs=0,cross_index=1, pre_train=None,temp_rate=1):

    if weights != 'imagenet':
        weight = None
    else:
        weight = weights

    spatial_a = InceptionSpatial_a(
                    n_neurons=n_neurons, seq_len=seq_len, classes=classes, 
                    weights=weight, dropout=dropout, fine=fine, retrain=False,
                    pre_file=pre_file,old_epochs=old_epochs,cross_index=cross_index)
    spatial_b = InceptionSpatial_b(n_neurons=n_neurons, seq_len=seq_len, classes=classes, 
                     weights=weight, dropout=dropout, fine=fine, retrain=False,
                     pre_file=pre_file,old_epochs=old_epochs,cross_index=cross_index,
                     input_shape=spatial_a.output_shape, input_tensor=spatial_a.output)

    # if (weights == 'pretrain') & (not retrain):
    #     print(glob.glob('weights/' + 'a_sinception_spatial2fc_256' + '-{:02d}-*.hdf5'.format(pre_train[0]))[-1])
    #     spatial_a.load_weights(glob.glob('weights/' + 'a_sinception_spatial2fc_256' + '-{:02d}-*.hdf5'.format(pre_train[0]))[-1])
    #     spatial_b.load_weights(glob.glob('weights/' + 'b_sinception_spatial2fc_256' + '-{:02d}-*.hdf5'.format(pre_train[0]))[-1])
    #     print ('load spatial weights')

    for layer in spatial_a.layers:
        layer.trainable = False

    temporal_a = InceptionTemporal_a(
                    n_neurons=n_neurons, seq_len=seq_len, classes=classes, 
                    weights=weight, dropout=dropout, fine=fine, retrain=False,
                    pre_file=pre_file,old_epochs=old_epochs,cross_index=cross_index)
    temporal_b = InceptionTemporal_b(
                    n_neurons=n_neurons, seq_len=seq_len, classes=classes, 
                    weights=weight, dropout=dropout, fine=fine, retrain=False,
                    pre_file=pre_file,old_epochs=old_epochs,cross_index=cross_index,
                    input_shape=temporal_a.output_shape, input_tensor=temporal_a.output)
    
    # if (weights == 'pretrain') & (not retrain):
    #     print(glob.glob('weights/' + 'a_sincept_temporal1_256' + '-{:02d}-*.hdf5'.format(pre_train[1]))[-1])
    #     temporal_a.load_weights(glob.glob('weights/' + 'a_sincept_temporal1_256' + '-{:02d}-*.hdf5'.format(pre_train[1]))[-1])
    #     temporal_b.load_weights(glob.glob('weights/' + 'b_sincept_temporal1_256' + '-{:02d}-*.hdf5'.format(pre_train[1]))[-1])
    #     print ('load temporal weights')

    for layer in temporal_a.layers:
        layer.trainable = False

    input1 = Input(shape=(seq_len, 299,299,3))
    input2 = Input(shape=(seq_len, 299,299,20))
    spatial = spatial_a(input1)
    temporal = temporal_a(input2)
    model_spatial_a = Model(inputs=input1, outputs=spatial)
    model_temporal_a = Model(inputs=input2, outputs=temporal)

    model_spatial_b = Model(inputs=input1, outputs=spatial_b(model_spatial_a.output))
    model_temporal_b = Model(inputs=input2, outputs=temporal_b(model_temporal_a.output))

    average = Lambda(foo,output_shape=foo_output_shape)([model_spatial_b.output, model_temporal_b.output])
    result_model = Model(inputs=[input1, input2], outputs=average)

    if retrain:
        print(glob.glob('weights/' + pre_file + '-{:02d}-*.hdf5'.format(old_epochs))[-1])
        result_model.load_weights(glob.glob('weights/' + pre_file + '-{:02d}-*.hdf5'.format(old_epochs))[-1])

    return result_model


def InceptionMultistream3(n_neurons=256, seq_len=3, classes=101, weights='imagenet', 
    dropout=0.8, fine=True, retrain=False, pre_file='',old_epochs=0,cross_index=1, pre_train=None,temp_rate=1):

    if weights != 'imagenet':
        weight = None
    else:
        weight = weights

    spatial_a = InceptionSpatial_a(
                    n_neurons=n_neurons, seq_len=seq_len, classes=classes, 
                    weights=weight, dropout=dropout, fine=fine, retrain=False,
                    pre_file=pre_file,old_epochs=old_epochs,cross_index=cross_index)
    spatial_b = InceptionSpatial_b(n_neurons=n_neurons, seq_len=seq_len, classes=classes, 
                     weights=weight, dropout=dropout, fine=fine, retrain=False,
                     pre_file=pre_file,old_epochs=old_epochs,cross_index=cross_index,
                     input_shape=spatial_a.output_shape, input_tensor=spatial_a.output)

    if (weights == 'pretrain') & (not retrain):
        print(glob.glob('weights/' + 'a_sinception_spatial2fc_256' + '-{:02d}-*.hdf5'.format(pre_train[0]))[-1])
        spatial_a.load_weights(glob.glob('weights/' + 'a_sinception_spatial2fc_256' + '-{:02d}-*.hdf5'.format(pre_train[0]))[-1])
        spatial_b.load_weights(glob.glob('weights/' + 'b_sinception_spatial2fc_256' + '-{:02d}-*.hdf5'.format(pre_train[0]))[-1])
        print ('load spatial weights')

    for layer in spatial_a.layers:
        layer.trainable = False

    temporal_a = InceptionTemporal_a(
                    n_neurons=n_neurons, seq_len=seq_len, classes=classes, 
                    weights=weight, dropout=dropout, fine=fine, retrain=False,
                    pre_file=pre_file,old_epochs=old_epochs,cross_index=cross_index)
    temporal_b = InceptionTemporal_b(
                    n_neurons=n_neurons, seq_len=seq_len, classes=classes, 
                    weights=weight, dropout=dropout, fine=fine, retrain=False,
                    pre_file=pre_file,old_epochs=old_epochs,cross_index=cross_index,
                    input_shape=temporal_a.output_shape, input_tensor=temporal_a.output)
    
    if (weights == 'pretrain') & (not retrain):
        print(glob.glob('weights/' + 'a_sincept_temporal1_256' + '-{:02d}-*.hdf5'.format(pre_train[1]))[-1])
        temporal_a.load_weights(glob.glob('weights/' + 'a_sincept_temporal1_256' + '-{:02d}-*.hdf5'.format(pre_train[1]))[-1])
        temporal_b.load_weights(glob.glob('weights/' + 'b_sincept_temporal1_256' + '-{:02d}-*.hdf5'.format(pre_train[1]))[-1])
        print ('load temporal weights')

    for layer in temporal_a.layers:
        layer.trainable = False

    input1 = Input(shape=(seq_len, 299,299,3))
    input2 = Input(shape=(seq_len, 299,299,20))
    spatial = spatial_a(input1)
    temporal = temporal_a(input2)
    model_spatial_a = Model(inputs=input1, outputs=spatial)
    model_temporal_a = Model(inputs=input2, outputs=temporal)
    concat = Lambda(concat_fusion, output_shape=concat_fusion_output_shape)([model_spatial_a.output, model_temporal_a.output])
    concat = Conv3D(2048, (1, 1, 1), activation='relu', padding='valid', name='conv3')(concat)

    #concat = spatial_b(concat)
    model_concat = Model(inputs=[input1, input2], outputs=concat)
    print(model_concat.output_shape)

    model_spatial_b = Model(inputs=[input1, input2], outputs=spatial_b(model_concat.output))
    model_temporal_b = Model(inputs=input2, outputs=temporal_b(model_temporal_a.output))
    print(model_spatial_b.output_shape)
    average = Lambda(foo,output_shape=foo_output_shape)([model_spatial_b.output, model_temporal_b.output])
    result_model = Model(inputs=[input1, input2], outputs=average)
    print(result_model.output_shape)

    if retrain:
        print(glob.glob('weights/' + pre_file + '-{:02d}-*.hdf5'.format(old_epochs))[-1])
        result_model.load_weights(glob.glob('weights/' + pre_file + '-{:02d}-*.hdf5'.format(old_epochs))[-1])

    return result_model

def train_process(model, pre_file, data_type, epochs=20, dataset='ucf101', 
    retrain=False, classes=101, cross_index=1, seq_len=3, old_epochs=0, batch_size=16, split_sequence=False, fine=True):

    out_file = r'{}database/{}-train{}-split{}-new.pickle'.format(data_output_path,dataset,seq_len,cross_index)
    valid_file = r'{}database/{}-test{}-split{}-test3.pickle'.format(data_output_path,dataset,seq_len,cross_index)

    with open(out_file,'rb') as f1:
        keys = pickle.load(f1)
    len_samples = len(keys)

    with open(valid_file,'rb') as f2:
        keys_valid = pickle.load(f2)
    len_valid = len(keys_valid)

    print('-'*40)
    print('{} training'.format(pre_file))
    print( 'Number samples: {}'.format(len_samples))
    print( 'Number valid: {}'.format(len_valid))
    print('-'*40)

    histories = []
    if server:
        steps = len_samples/batch_size
        validation_steps = int(np.ceil(len_valid*1.0/batch_size))
    else:
        steps = len_samples/batch_size
        validation_steps = int(np.ceil(len_valid*1.0/batch_size))

    lrate = LearningRateScheduler(step_decay)
    filepath="weights/" + pre_file + "-{epoch:02d}-{val_acc:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='max')
    csv_logger = CSVLogger('histories/{}_{}_{}_{}e_cr{}.csv'.format(pre_file,seq_len,old_epochs,epochs,cross_index), append=True, separator=';')
    callbacks_list = [checkpoint, csv_logger]
    
    for e in range(epochs):
        print('Epoch', old_epochs+e+1)
        random.shuffle(keys)
        model.fit_generator(
            gd.getTrainData(
                keys=keys,batch_size=batch_size,dataset=dataset,
                classes=classes,train='train',data_type=data_type,split_sequence=split_sequence,epochs=1), 
            verbose=1, 
            max_queue_size=20, 
            steps_per_epoch=steps, 
            epochs=1,
            shuffle=True,
            validation_data=gd.getTrainData(
                keys=keys_valid,batch_size=batch_size,dataset=dataset,classes=classes,train='valid',data_type=data_type,split_sequence=split_sequence),
            validation_steps=validation_steps,
	    callbacks=callbacks_list
        )

def test_process_25(model, pre_file, data_type, epochs=20, dataset='ucf101', 
    classes=101, cross_index=1, seq_len=3, batch_size=1, split_sequence=False):

    print(glob.glob('weights/' + pre_file + '-{:02d}-*.hdf5'.format(epochs))[0])
    model.load_weights(glob.glob('weights/' + pre_file + '-{:02d}-*.hdf5'.format(epochs))[0])

    out_file = r'{}database/{}-test{}-split{}-test25.pickle'.format(data_output_path,dataset,seq_len,cross_index)
    with open(out_file,'rb') as f2:
        keys = pickle.load(f2)
    # keys = keys[520:525]
    len_samples = len(keys)

    print('-'*40)
    print('{} testing'.format(pre_file))
    print ('Number samples: {}'.format(len_samples))
    print('-'*40)

    Y_test = gd.getClassData(keys)
    steps = int(np.ceil(len_samples*1.0/batch_size))

    time_start = time.time()

    y = model.predict_generator(
        gd.getTrainData(
            keys=keys,batch_size=batch_size,dataset=dataset,classes=classes,train='test',data_type=data_type,split_sequence=split_sequence), 
        max_queue_size=20, 
        steps=steps)

    run_time = time.time() - time_start

    y_ = y[2]
    y_pred = []
    for i in range(0, 10*len_samples, 10):
        temp = np.sum(y_[i:i+9], axis=0)*1.0/10
        y_pred.append(temp)

    with open('results/{}_{}e_cr{}.pickle'.format(pre_file,epochs,cross_index),'wb') as fw3:
        pickle.dump([y_pred, Y_test],fw3)

    print(np.array(y_pred).shape)
    y_classes = np.array(y_pred).argmax(axis=-1)
    print(classification_report(Y_test, y_classes, digits=6))
    print(accuracy_score(Y_test, y_classes))
    print(accuracy_score(Y_test, y_classes, normalize=False))
    print ('Run time: {}'.format(run_time))

def test_process(model, pre_file, data_type, epochs=20, dataset='ucf101', 
    classes=101, cross_index=1, seq_len=3, batch_size=1, split_sequence=False):

    print(glob.glob('weights/' + pre_file + '-{:02d}-*.hdf5'.format(epochs))[0])
    model.load_weights(glob.glob('weights/' + pre_file + '-{:02d}-*.hdf5'.format(epochs))[0])

    y_pred = np.zeros((3783, 101))

    for i in range(9):
        out_file = r'{}database/{}-test{}-split{}-test0'.format(data_output_path,dataset,seq_len,cross_index)+str(i)+'.pickle'
        with open(out_file,'rb') as f2:
            keys = pickle.load(f2)
        len_samples = len(keys)

        print('-'*40)
        print('{} testing'.format(pre_file))
        print ('Number samples: {}'.format(len_samples))
        print('-'*40)

        Y_test = gd.getClassData(keys)
        steps = int(np.ceil(len_samples*1.0/batch_size))

        time_start = time.time()

        y_ = model.predict_generator(
            gd.getTrainData(
                keys=keys,batch_size=batch_size,dataset=dataset,classes=classes,train='test',data_type=data_type,split_sequence=split_sequence), 
            max_queue_size=20, 
            steps=steps)

        run_time = time.time() - time_start

        #y_ = y[2]
        y_p = []
        for i in range(0, 10*len_samples, 10):
            temp = np.sum(y_[i:i+9], axis=0)*1.0/10
            y_p.append(temp)
        y_pred = y_pred + y_p

    with open('results/{}_{}e_cr{}.pickle'.format(pre_file,epochs,cross_index),'wb') as fw3:
        pickle.dump([y_pred, Y_test],fw3)

    print(np.array(y_pred).shape)
    y_classes = np.array(y_pred).argmax(axis=-1)
    print(classification_report(Y_test, y_classes, digits=6))
    confusion_mtx = confusion_matrix(Y_test, y_classes)
    np.savetxt('results/{}_{}e_cr{}.csv'.format(pre_file,epochs,cross_index), confusion_mtx, delimiter=",") 
    print ('Run time: {}'.format(run_time))
