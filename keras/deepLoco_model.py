# coding=utf-8

import numpy as np
from keras.layers import Conv2D, Dense, Reshape, Conv1D, LeakyReLU
from keras.layers import MaxPooling2D, MaxPooling1D, BatchNormalization, Flatten, Activation
from keras.layers.merge import concatenate, add
from keras import losses
from keras.callbacks import Callback
import tensorflow as tf
from keras import backend as K
from keras.models import *
from keras.optimizers import *


# define a function that projects an image to the range [0,1]
def project_01(im):
    im = np.squeeze(im)
    min_val = im.min()
    max_val = im.max()
    return (im - min_val)/(max_val - min_val)


# normalize image given mean and std
def normalize_im(im, dmean, dstd):
    im = np.squeeze(im)
    im_norm = np.zeros(im.shape,dtype=np.float32)
    im_norm = (im - dmean)/dstd
    return im_norm


def lap_kernel(x,y,sigma):
    # x,y - 2/3D position. shape=(batchsize, sample_len, 2)
    # sigma - often choose to be near the expected localization precision do the system, ie., 20 to 50 nanometers
    # output K(x,y) = 1/(2*sigma)*exp(-||x-y||/sigma)
    # a=K.expand_dims(x,2)
    # print(a.shape)
    # b=K.repeat_elements(a,256,axis=2)
    # print(b.shape)

    matrix1 = K.repeat_elements(K.expand_dims(x,2),256,axis=2) # repeat column
    matrix2 = K.repeat_elements(K.expand_dims(y,1),256,axis=1)
    return K.exp(-K.sum(K.abs(matrix1-matrix2), axis=3)/sigma)


def deepLoco_absolute_error(y_true, y_pred):

    weight_true = y_true[:,:,0]
    position_true = y_true[:,:,1:]
    weight_pred = y_pred[:,:,0]
    position_pred = y_pred[:,:,1:]

    error = K.abs(K.batch_dot(weight_true,position_true)-K.batch_dot(weight_pred,position_pred))
    return K.sum(error)

dle = DLE = deepLoco_absolute_error
# batch_size = 8
# num_positions = 10
#
# y_true = 10 * np.random.rand(batch_size, num_positions, 3)
# y_pred = 10 * np.random.rand(batch_size, num_positions, 3)
#
# x=K.variable(y_true)
# y=K.variable(y_pred)
# loss = deepLoco_absolute_error(x,y)
# print(K.eval(loss))


def deepLoco_loss(y_true, y_pred):

    weight_true = y_true[:,:,0]
    position_true = y_true[:,:,1:]
    weight_pred = y_pred[:,:,0]
    position_pred = y_pred[:,:,1:]

    loss = K.variable(value=0)
    for sigma in range(20,50,10):
        K_tt = lap_kernel(position_true,position_true,sigma=sigma)
        K_tp = lap_kernel(position_true,position_pred,sigma=sigma)
        K_pp = lap_kernel(position_pred,position_pred,sigma=sigma)
        # print(K_tt.shape, K_tp.shape, K_pp.shape)

        s1 = K.batch_dot(K.batch_dot(K.expand_dims(weight_true,axis=1),K_tt),K.expand_dims(weight_true,axis=2))
        s2 = K.batch_dot(K.batch_dot(K.expand_dims(weight_true,axis=1),K_tp),K.expand_dims(weight_pred,axis=2))
        s3 = K.batch_dot(K.batch_dot(K.expand_dims(weight_pred,axis=1),K_pp),K.expand_dims(weight_pred,axis=2))
        print(s1.shape, s2.shape, s3.shape)

        loss = loss + K.sum(s1-2*s2+s3)
    # print(loss.shape)
    return loss


# Define the loss history recorder
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def add_common_layers(y):
        y = BatchNormalization()(y)
        y = LeakyReLU()(y)

        return y


def deepLoco(inputs):

    print("input shape:",inputs.shape)
    conv1 = Conv2D(16, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    print ("conv1 shape:",conv1.shape)
    conv1 = Conv2D(16, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    print ("conv1 shape:",conv1.shape)
    conv1 = Conv2D(64, 5, activation = 'relu', strides= 2, padding = 'same', kernel_initializer = 'he_normal')(conv1)
    print ("conv2 shape:",conv1.shape)
    # pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # print ("pool1 shape:",pool1.shape)

    conv2 = Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    print ("conv2 shape:",conv2.shape)
    conv2 = Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    print ("conv2 shape:",conv2.shape)
    conv2 = Conv2D(256, 3, activation = 'relu',  strides=2, padding = 'same', kernel_initializer = 'he_normal')(conv2)
    print ("conv3 shape:",conv2.shape)
    # pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # print ("pool2 shape:",pool2.shape)

    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    print ("conv3 shape:",conv3.shape)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    print ("conv3 shape:",conv3.shape)
    conv3 = Conv2D(256, 3, activation = 'relu', strides=4, padding = 'same', kernel_initializer = 'he_normal')(conv3)
    print ("conv3 shape:",conv3.shape)
    # pool3 = MaxPooling2D(pool_size=(4, 4))(conv3)
    # print ("pool3 shape:",pool3.shape)

    flat1 = Flatten()(conv3)
    print("flat ", flat1.shape)
    dense1 = Dense(2048)(flat1)
    print("dense1 ",dense1.shape)

    reshape1 = Reshape([2048, 1])(dense1)
    print("reshape1", reshape1.shape)

    shortcut = reshape1
    # res1 = build_resnet(reshape1, basic_block, [2])
    res1 = Conv1D(1, kernel_size=3, strides=1, padding='same')(reshape1)
    print("res1 ", res1.shape)
    res1 = LeakyReLU()(res1)
    res1 = BatchNormalization()(res1)
    # res1 = add_common_layers(res1)
    add1 = add([shortcut,res1])
    print("add1 ",add1.shape)
    add1 = LeakyReLU()(add1)
    add1 = BatchNormalization()(add1)

    shortcut = add1
    res2 = Conv1D(1, kernel_size=3, strides=1, padding='same')(add1)
    res2 = LeakyReLU()(res2)
    res2 = BatchNormalization()(res2)
    # res2 = add_common_layers(res2)
    print("res2 ",res2.shape)
    add2 = add([shortcut,res2])
    print("add2 ",add2.shape)
    add2 = LeakyReLU()(add2)
    add2 = BatchNormalization()(add2)

    weights = Conv1D(1, kernel_size=3, strides=8, padding='same')(add2)
    # weights = MaxPooling1D(pool_size=8)(weights)
    weights = Activation("relu")(weights)
    print("weights ", weights.shape)
    # positions = Dense(kernel_initializer="he_normal",
    #                   activation="softmax")(add2)
    # positions = Dense(units = 16)(add2)
    # print(positions.shape)
    # positions = Reshape([2048,16,1])(positions)

    positions = Conv1D(2, kernel_size=3, strides=8, padding='same')(add2)
    # positions = MaxPooling1D(pool_size=8)(positions)
    positions = Activation("sigmoid")(positions)
    print("positions ",positions.shape)

    # return [weights, positions]
    return concatenate([weights,positions],axis=2)


def build_model(input_dim):

    input = Input(shape=input_dim)

    combined_weight = deepLoco(input)
    print(combined_weight.shape)
    model = Model(inputs = input, outputs = combined_weight)
    model.compile(optimizer = Adam(lr = 1e-3), loss = [deepLoco_loss], metrics = [dle])

    # weights, positions = deepLoco(inputs)
    # model = Model(input = inputs, output = [weights, positions])
    # model.compile(optimizer = Adam(lr = 1e-3), loss = deepLoco_loss(input_dim), metrics = ['accuracy'], loss_weights=[0.0, 0.0, 1.0])

    return model

