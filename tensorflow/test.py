from deepLoco.deepLoco_bn import deepLoco_net
from deepLoco.trainer import Trainer_bn

from deepLoco import image_util
from deepLoco import util

import scipy.io as sio
import numpy as np
import os
import h5py
from sklearn.model_selection import train_test_split

####################################################
####             PREPARE WORKSPACE               ###
####################################################

# here indicating the GPU you want to use. if you don't have GPU, just leave it.
gpu_vis = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_vis; # 0,1,2,3

# here specify the path of the model you want to load
gpu_ind = '0'
model_path = 'gpu' + gpu_ind + '/models/60099_cpkt/models/final/model.cpkt'

data_channels = 1
np.random.seed(123)

####################################################
####                 FUNCTIONS                   ###
####################################################

# make the data a 4D vector
def preprocess(data, channels):
	nx = data.shape[1]
	ny = data.shape[2]
	return data.reshape((-1, nx, ny, channels))

####################################################
####                lOAD MODEL                   ###
####################################################

# set up args for the unet, should be exactly the same as the loading model
kwargs = {
    "layers": 5,
    "conv_times": 2,
    "features_root": 64,
    "filter_size": 3,
    "pool_size": 2,
    "summaries": True
}

net = deepLoco_net(img_channels=data_channels, cost="deepLoco_error", **kwargs)


####################################################
####                lOAD TRAIN                   ###
####################################################

# data_path = "data/TrainingSet_deepLoco_64.mat"
data_path = "data/TrainingSet_iso_du.mat"
file_path = "l_run/"

matfile = h5py.File(data_path, 'r')
data = np.array(matfile['images'])
positions = np.array(matfile['positions'])
weights = np.array(matfile['weights'])

X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(data, positions, weights, test_size=0.25, random_state=42)
print('Number of Training Examples: %d' % X_train.shape[0])
print('Number of Validation Examples: %d' % X_test.shape[0])
# mdict = {"X_test": X_test, "y_test": y_test, "w_test": w_test}
# sio.savemat("data/TrainingSet_iso_du_test", mdict)
# mdict = {"X_train": X_train, "y_train": y_train, "w_train": w_train}
# sio.savemat("data/TrainingSet_iso_du_train", mdict)


# Setting type
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')
w_train = w_train.astype('float32')
w_test = w_test.astype('float32')

#-- Training Data --#
max_train = X_train.max()
min_train = X_train.min()
X_train_norm = (X_train-min_train)/(max_train-min_train)

#-- Validating Data --#
max_test = X_test.max()
min_test = X_test.min()
X_test_norm = (X_test-min_test)/(max_test-min_test)

# patch size
psize =  X_train_norm.shape[1]

# Reshaping
X_train_norm = X_train_norm.reshape(X_train.shape[0], psize, psize, 1)
X_test_norm = X_test_norm.reshape(X_test.shape[0], psize, psize, 1)

# normalizing position label
y_train = y_train/psize
y_test = y_test/psize

Y_train = np.zeros([y_train.shape[0],y_train.shape[1],y_train.shape[2]+1])
Y_train[:,:,0] = w_train
Y_train[:,:,1:] = y_train
Y_test = np.zeros([y_test.shape[0],y_test.shape[1],y_test.shape[2]+1])
Y_test[:,:,0] = w_test
Y_test[:,:,1:] = y_test

data_provider = image_util.SimpleDataProvider(X_train_norm, Y_train)
valid_provider = image_util.SimpleDataProvider(X_test_norm, Y_train)


savename = "recovery_test"
predicts = []

valid_x, valid_y = valid_provider('full')
num = valid_x.shape[0]

# for i in range(num):
for i in range(min(50,num)):

    print('')
    print('')
    print('************* {} *************'.format(i))
    print('')
    print('')

    x_train, y_train = data_provider(23)
    x_input = valid_x[i:i+1,:,:,:]
    x_input = np.concatenate((x_input, x_train), axis=0)
    predict = net.predict(file_path, x_input, 1, True)
    predicts.append(predict[0:1,:,:])

predicts = np.concatenate(predicts, axis=0)
# util.save_mat(predicts, 'test{}Noise.mat'.format(level))
mdict = {"Predictions": predicts}
sio.savemat(file_path + savename + '.mat', mdict)




