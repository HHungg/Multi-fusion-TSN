import random
import numpy as np
import sys
import pickle
import random
from PIL import Image
import cv2
from keras.utils import np_utils
import config
from sklearn.metrics import classification_report
from keras import backend as K
import math

server = config.server()
data_output_path = config.data_output_path()
data_folder_seq = r'{}seq3/'.format(data_output_path)

def getTrainData(keys,batch_size,dataset,classes,train,data_type,split_sequence=False, epochs=100): 
    """
    mode 1: Single Stream
    mode 2: Two Stream
    mode 3: Multiple Stream
    """
    mode = len(data_type)
    
    while 1:
        for i in range(0, len(keys), batch_size):
            if mode == 1:
                X_train, Y_train = stack_single_sequence(
                    chunk=keys[i:i+batch_size],
                    data_type=data_type,
                    dataset=dataset,
                    train=train)
            else:
                X_train, Y_train = stack_multi_sequence(
                    chunk=keys[i:i+batch_size],
                    multi_data_type=data_type,
                    dataset=dataset,
                    train=train)
            if(train=='test'):
                Y_train = np.repeat(Y_train, 10)
            Y_train = np_utils.to_categorical(Y_train,classes)
            if (train != 'train') & (i != 0) & ((i/batch_size) % 50 == 0):
                print ('Test batch {}'.format(i/batch_size+1))
            if(mode==1):
                yield X_train, [np.array(Y_train), np.array(Y_train), np.array(Y_train)]
            else:
                yield X_train, np.array(Y_train)

def stack_single_sequence(chunk,data_type,dataset,train):
    size = random_size()
    mode_crop = random.randint(0, 1)
    flip = random.randint(0, 2)
    mode_corner_crop = random.randint(0, 5)

    if train != 'train':
        size = 224
    if dataset == 'ucf101':
        x = random.randint(0, 340-size)
        y = random.randint(0, 256-size)
        if train != 'train':
            x = (340-size)/2
            y = (256-size)/2
    else:
        x = -1
        y = -1

    pre_random = [size, mode_crop, flip, mode_corner_crop, x, y]

    labels = []
    stack_return = []
    if data_type[0] == 0:
        for rgb in chunk:
            labels.append(rgb[2])
            if (train == 'train'):
                render_rgb = random_position(rgb[4], 3)
                stack_return.append(stack_seq_rgb(rgb[0],render_rgb,pre_random,dataset,train))
            elif (train == 'test'):
                render_rgb = rgb[1]
                for i in range(0,5):
                    for j in range(0,2):
                        stack_return.append(stack_test_rgb(rgb[0],render_rgb,i,j,dataset,train))
                        # return np.array(stack_return). labels
            else:
                render_rgb = rgb[1]
                stack_return.append(stack_seq_rgb(rgb[0],render_rgb,pre_random,dataset,train))
            
    else:
        for opt in chunk:
            labels.append(opt[2])
            if (train == 'train'):
                render_opt = random_position(opt[4], 3, False, data_type[0])
                stack_return.append(stack_seq_optical_flow(opt[0],render_opt,data_type[0],pre_random,dataset,train))
            elif (train == 'test'):
                render_opt = opt[1]
                for i in range(0,5):
                    for j in range(0,2):
                        stack_return.append(stack_test_optical_flow(opt[0],render_opt,data_type[0],i,j,dataset,train))
            else:
                render_opt = opt[1]
                stack_return.append(stack_seq_optical_flow(opt[0],render_opt,data_type[0],pre_random,dataset,train))

    if len(stack_return) < len(chunk):
        print ('Stacked data error')
        sys.exit()

    return np.array(stack_return), labels

def stack_multi_sequence(chunk,multi_data_type,dataset,train):
    size = random_size()
    mode_crop = random.randint(0, 1)
    flip = random.randint(0, 2)
    mode_corner_crop = random.randint(0, 5)

    if train != 'train':
        size = 224
    if dataset == 'ucf101':
        x = random.randint(0, 340-size)
        y = random.randint(0, 256-size)
        if train != 'train':
            x = (340-size)/2
            y = (256-size)/2
    else:
        x = -1
        y = -1

    pre_random = [size, mode_crop, flip, mode_corner_crop, x, y]

    labels = []
    stack_return_rgb = []
    stack_return_opt = []

    for rgb in chunk:
        labels.append(rgb[2])
        if (train == 'train'):
            render_opt = random_position(rgb[4] - 10 * multi_data_type[1], 3, False, multi_data_type[1])
            render_rgb = np.array(render_opt) + 10 * multi_data_type[1]
            stack_return_rgb.append(stack_seq_rgb(rgb[0],render_rgb,pre_random,dataset,train))
            stack_return_opt.append(stack_seq_optical_flow(rgb[0],render_opt,multi_data_type[1],pre_random,dataset,train))
        elif (train == 'test'):
            render_opt = rgb[1]
            render_rgb = render_opt
            for i in range(0,5):
                for j in range(0,2):
                    stack_return_rgb.append(stack_test_rgb(rgb[0],render_rgb,i,j,dataset,train))
                    stack_return_opt.append(stack_test_optical_flow(rgb[0],render_opt,multi_data_type[1],i,j,dataset,train))
        else:
            render_opt = rgb[1]
            render_rgb = render_opt
            stack_return_rgb.append(stack_seq_rgb(rgb[0],render_rgb,pre_random,dataset,train))
            stack_return_opt.append(stack_seq_optical_flow(rgb[0],render_opt,multi_data_type[1],pre_random,dataset,train))

    if len(stack_return_rgb) < len(chunk):
        print ('Stacked data error')
        sys.exit()

    return [np.array(stack_return_rgb), np.array(stack_return_opt)], labels

def stack_seq_rgb(path_video,render_rgb,pre_random,dataset,train):
    return_stack = []
    data_folder_rgb = r'{}{}-rgb/'.format(data_output_path,dataset)

    name_video = path_video.split('/')[1]

    size = pre_random[0]
    mode_crop = pre_random[1]
    flip = pre_random[2]
    mode_corner_crop = pre_random[3]
    x = pre_random[4]
    y = pre_random[5]

    hx = 256
    wx = 340

    for i in render_rgb:
        i_index = 'frame' + str(i).zfill(6) + '.jpg'

        rgb = cv2.imread(data_folder_rgb + name_video + '/' + i_index)
        if rgb is None:
            print ('Not found: ' + data_folder_rgb + name_video + '/' + i_index)
            sys.exit()

        if x == -1:
            hx, wx, cx = rgb.shape
            x = random.randint(0, wx-size)
            y = random.randint(0, hx-size)
            if train != 'train':
                x = (wx-size)/2
                y = (hx-size)/2

        if train == 'train':
            rgb = random_corner_crop(rgb, size, mode_corner_crop, wx, hx)
            rgb = random_flip(rgb, size, flip)
        else:
            rgb = image_crop(rgb, x, y, size)

        height, width, channel = rgb.shape
        if height == size:
            # if size != 224:
            rgb = cv2.resize(rgb, (299, 299))   
            # print size
            rgb = rgb.astype('float16',copy=False)
            rgb_nor = rgb/255
            # rgb-=0.5
            # rgb*=2
            # rgb_nor = rgb - rgb.mean()
        else:
            print(mode_crop, flip, mode_corner_crop, size, height, x, y)
            sys.exit()

        # return_stack.append(rgb)
        return_stack.append(rgb_nor)
    return np.array(return_stack)

def stack_test_rgb(path_video,render_rgb,mode_corner_crop,flip,dataset,train):
    return_stack = []
    data_folder_rgb = r'{}{}-rgb/'.format(data_output_path,dataset)
    size = 224

    name_video = path_video.split('/')[1]

    hx = 256
    wx = 340

    for i in render_rgb:
        if(i<=0):
            i = 1
        i_index = 'frame' + str(i).zfill(6) + '.jpg'
        rgb = cv2.imread(data_folder_rgb + name_video + '/' + i_index)
        if rgb is None:
            print ('Not found: ' + data_folder_rgb + name_video + '/' + i_index)
            sys.exit()

        rgb = random_corner_crop(rgb, size, mode_corner_crop, wx, hx)
        rgb = random_flip(rgb, size, flip)

        height, width, channel = rgb.shape
        if height == size:
            # if size != 224:
            rgb = cv2.resize(rgb, (299, 299))   
            # print size
            rgb = rgb.astype('float16',copy=False)
            rgb_nor = rgb/255
            # rgb-=0.5
            # rgb*=2
            # rgb_nor = rgb - rgb.mean()
        else:
            print(flip, mode_corner_crop, size, height)
            sys.exit()

        # return_stack.append(rgb)
        return_stack.append(rgb_nor)
    return np.array(return_stack)

def stack_seq_optical_flow(path_video,render_opt,data_type,pre_random,dataset,train):
    data_folder_opt = r'{}{}-opt{}/'.format(data_output_path,dataset,data_type)
    name_video = path_video.split('/')[1]

    if(data_type == 1):
        u = data_folder_opt + 'u/' + name_video + '/frame'
        v = data_folder_opt + 'v/' + name_video + '/frame'
    else:
        name_class = name_video.split('_')[1]
        u = data_folder_opt + 'u/' + name_class + '/' + name_video + '/frame'
        v = data_folder_opt + 'v/' + name_class + '/' + name_video + '/frame'

    # print (u,v)

    hx = 256
    wx = 340

    return_data = []

    size = pre_random[0]
    mode_crop = pre_random[1]
    flip = pre_random[2]
    mode_corner_crop = pre_random[3]
    x = pre_random[4]
    y = pre_random[5]

    if (render_opt[0] <= 0):
        render_opt[0] = 1
    render = render_opt

    len_render_opt = len(render)

    for k in range(len_render_opt):
        nstack = np.zeros((256,340,20))
        if (render[k] <= data_type):
            render[k] = data_type
        for i in range(10):
            img_u = cv2.imread(u + str(render[k]/data_type + i).zfill(6) + '.jpg', 0)
            img_v = cv2.imread(v + str(render[k]/data_type + i).zfill(6) + '.jpg', 0) 
            if img_u is None:
                print ('Not found:' + u + str(render[k]/data_type + i).zfill(6) + '.jpg')
                img_u = nstack[:,:,2*(i-1)]
                img_v = nstack[:,:,2*(i-1)+1]
            if(data_type==3):
                img_u = cv2.resize(img_u,(340,256),interpolation=cv2.INTER_CUBIC)
                img_v = cv2.resize(img_v,(340,256),interpolation=cv2.INTER_CUBIC)
            img_u = img_u[:,0:340]
            img_v = img_v[:,0:340]
            nstack[:,:,2*i] = img_u
            nstack[:,:,2*i+1] = img_v

        if train == 'train':
            nstack = random_corner_crop(nstack, size, mode_corner_crop, wx, hx)
            nstack = random_flip(nstack, size, flip)
        else:
            nstack = image_crop(nstack, x, y, size)

        height, width, channel = nstack.shape
        if (height == size) & (width == size):
            nstack = cv2.resize(nstack, (299, 299))
            # print size
            nstack_nor = nstack.astype('float16',copy=False)
            nstack_nor/=255
            # nstack_nor = nstack - nstack.mean(axis=2, keepdims=True)
            # nstack_nor = nstack - nstack.mean()
        else:
            print('Error', mode_crop, flip, mode_corner_crop, size, height, x, y)
            sys.exit()

        return_data.append(nstack_nor)

    if (len_render_opt == 1):
        return_data.append(nstack_nor)

    return return_data

def stack_test_optical_flow(path_video,render_opt,data_type,mode_corner_crop,flip,dataset,train):
    data_folder_opt = r'{}{}-opt{}/'.format(data_output_path,dataset,data_type)
    name_video = path_video.split('/')[1]

    if(data_type == 1):
        u = data_folder_opt + 'u/' + name_video + '/frame'
        v = data_folder_opt + 'v/' + name_video + '/frame'
    else:
        name_class = name_video.split('_')[1]
        u = data_folder_opt + 'u/' + name_class + '/' + name_video + '/frame'
        v = data_folder_opt + 'v/' + name_class + '/' + name_video + '/frame'

    # print (u,v)

    hx = 256
    wx = 340

    return_data = []
    size = 224

    if (render_opt[0] <= 0):
        render_opt[0] = 1
    render = render_opt

    len_render_opt = len(render)

    for k in range(len_render_opt):
        nstack = np.zeros((256,340,20))
        if (render[k] <= data_type):
            render[k] = data_type
        for i in range(10):
            img_u = cv2.imread(u + str(render[k]/data_type + i).zfill(6) + '.jpg', 0)
            img_v = cv2.imread(v + str(render[k]/data_type + i).zfill(6) + '.jpg', 0) 
            if img_u is None:
                print ('Not found:' + u + str(render[k]/data_type + i).zfill(6) + '.jpg')
                img_u = nstack[:,:,2*(i-1)]
                img_v = nstack[:,:,2*(i-1)+1]
            if(data_type==3):
                img_u = cv2.resize(img_u,(340,256),interpolation=cv2.INTER_CUBIC)
                img_v = cv2.resize(img_v,(340,256),interpolation=cv2.INTER_CUBIC)
            img_u = img_u[:,0:340]
            img_v = img_v[:,0:340]
            nstack[:,:,2*i] = img_u
            nstack[:,:,2*i+1] = img_v

        nstack = random_corner_crop(nstack, size, mode_corner_crop, wx, hx)
        nstack = random_flip(nstack, size, flip)

        height, width, channel = nstack.shape
        if (height == size) & (width == size):
            nstack = cv2.resize(nstack, (299, 299))
            # print size
            nstack_nor = nstack.astype('float16',copy=False)
            nstack_nor/=255
            # nstack_nor = nstack - nstack.mean(axis=2, keepdims=True)
            # nstack_nor = nstack - nstack.mean()
        else:
            print('Error', flip, mode_corner_crop, size, height,wx, hx)
            sys.exit()

        return_data.append(nstack_nor)

    if (len_render_opt == 1):
        return_data.append(nstack_nor)

    return return_data

def getClassData(keys,cut=0):
    labels = []
    if cut == 0:
        for opt in keys:
            labels.append(opt[2])
    else:
        i = 0
        for opt in keys:
            labels.append(opt[2])
            i += 1
            if i >= cut:
                break

    return labels

def convert_weights(weights, depth, size=3, ins=32):
    mat = weights[0]
    mat2 = np.empty([size,size,depth,ins])
    for i in range(ins):
        x=(mat[:,:,0,i] + mat[:,:,1,i] + mat[:,:,2,i])/3
        for j in range(depth):
            mat2[:,:,j,i] = x
    return [mat2]

def random_position(length, num_seq, rgb=True, data_type=1):
    length = length - 1
    divide = length / num_seq
    train_render = []
    if (rgb):
        for i in range(num_seq):
            if i < num_seq - 1:
                if(i==0):
                    k = np.random.randint(1,divide*(i+1)+1)
                else:
                    k = np.random.randint(divide*i+1,divide*(i+1)+1)
            else:
                k = np.random.randint(divide*i+1,length+1)
            train_render.append(k)
    else:
        if length > 30*data_type:
            for i in range(num_seq):
                if i < num_seq - 1:
                     if(i==0):
                         k = np.random.randint(1,divide*(i+1)-9*data_type+1)
                     else:
                         k = np.random.randint(divide*i+1,divide*(i+1)-9*data_type+1)
                else:
                    k = np.random.randint(divide*i+1,length-9*data_type+1)
                train_render.append(k)
        elif (length > 10*data_type):
            for i in range(num_seq):
                k = np.random.randint(1, length-9*data_type)
                train_render.append(k)
                train_render.sort()
        else:
            train_render = np.ones((num_seq,), dtype=int)
    return train_render

def random_size():
    size = [256,224,192,168]
    return random.choice(size)

def random_flip(image, size, flip):
    image_flip = image.copy()
    if (flip==1):
        image_flip = cv2.flip(image_flip, 1)
    return image_flip

def random_crop(image, size, mode_crop, mode_corner_crop, x, y,w=340,h=256):
    if mode_crop == 0:
        return random_corner_crop(image, size, mode_corner_crop,w,h)
    else:
        return image_crop(image, x, y, size)

def random_corner_crop(image, size, mode_corner_crop,w=340,h=256):
    if mode_corner_crop == 0:
        return image_crop(image, 0, 0, size)
    elif mode_corner_crop == 1:
        return image_crop(image, w-size, 0, size)
    elif mode_corner_crop == 2:
        return image_crop(image, 0, h-size, size)
    elif mode_corner_crop == 3:
        return image_crop(image, w-size, h-size, size)
    else:
        return image_crop(image, (w-size)/2, (h-size)/2, size)
       
def image_crop(image, x, y, size):
    return image[y:y+size,x:x+size,:]
