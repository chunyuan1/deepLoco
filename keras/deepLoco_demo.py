# coding=utf-8

import h5py
import time
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from deepLoco_model import project_01, normalize_im, build_model, LossHistory
import tensorflow as tf

np.random.seed(123)


# initialize the number of epochs to trainer for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 75
INIT_LR = 1e-3
BS = 16

# Load data
data_path = "data/TrainingSet_deepLoco_64.mat"
# data_path = "data/TrainingSet_iso.mat"
# data_path = "data/TrainingSet_deepLoco_iso_64.mat"
# data_path = "data/TrainingSet_iso_du.mat"
file_path = "k_run/"
meanstd_file = "meanstd.mat"
weights_file = "weights.hdf5"
test_file = "TestSet.mat"

matfile = h5py.File(data_path, 'r')
data = np.array(matfile['images'])
positions = np.array(matfile['positions'])
weights = np.array(matfile['weights'])

# group weights and position to form training labels
# positions = positions.tolist()
# weights = weights.tolist()
# labels = [[weight]+[position] for weight,position in zip(weights,positions)]

X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(data, positions, weights, test_size=0.25, random_state=42)
print('Number of Training Examples: %d' % X_train.shape[0])
print('Number of Validation Examples: %d' % X_test.shape[0])
# mdict = {"X_test": X_test, "y_test": y_test, "w_test": w_test}
# sio.savemat("data/TrainingSet_iso_test", mdict)
# mdict = {"X_train": X_train, "y_train": y_train, "w_train": w_train}
# sio.savemat("data/TrainingSet_iso_train", mdict)


# Setting type
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')
w_train = w_train.astype('float32')
w_test = w_test.astype('float32')

max_train = X_train.max()
min_train = X_train.min()
X_train_norm = (X_train-min_train)/(max_train-min_train)

# # Training set normalization
# mean_train = np.zeros(X_train.shape[0], dtype=np.float32)
# std_train = np.zeros(X_train.shape[0], dtype=np.float32)
# for i in range(X_train.shape[0]):
#     X_train[i, :, :] = project_01(X_train[i, :, :])
#     mean_train[i] = X_train[i, :, :].mean()
#     std_train[i] = X_train[i, :, :].std()
#
# # resulting normalized training images
# mean_val_train = mean_train.mean()
# std_val_train = std_train.mean()
# X_train_norm = np.zeros(X_train.shape, dtype=np.float32)
# for i in range(X_train.shape[0]):
#     X_train_norm[i, :, :] = normalize_im(X_train[i, :, :], mean_val_train, std_val_train)

max_test = X_test.max()
min_test = X_test.min()
X_test_norm = (X_test-min_test)/(max_test-min_test)

# # Test set normalization
# mean_test = np.zeros(X_test.shape[0],dtype=np.float32)
# std_test = np.zeros(X_test.shape[0], dtype=np.float32)
# for i in range(X_test.shape[0]):
#     X_test[i, :, :] = project_01(X_test[i, :, :])
#     mean_test[i] = X_test[i, :, :].mean()
#     std_test[i] = X_test[i, :, :].std()
#
# # resulting normalized test images
# mean_val_test = mean_test.mean()
# std_val_test = std_test.mean()
# X_test_norm = np.zeros(X_test.shape, dtype=np.float32)
# for i in range(X_test.shape[0]):
#     X_test_norm[i, :, :] = normalize_im(X_test[i, :, :], mean_val_test, std_val_test)

# patch size
psize =  X_train_norm.shape[1]

# Reshaping
X_train_norm = X_train_norm.reshape(X_train.shape[0], psize, psize, 1)
X_test_norm = X_test_norm.reshape(X_test.shape[0], psize, psize, 1)

# normalizing position label
y_train = y_train/psize
y_test = y_test/psize

# Reshaping labels
# Y_train = y_train.reshape(y_train.shape[0], psize, y_train.shape[2], 1)
# Y_test = y_test.reshape(y_test.shape[0], psize, y_train.shape[2], 1)
# Y_train = [np.concatenate((np.reshape(weight,(-1,1)),position),axis=1) for weight,position in zip(w_train,y_train)]
# Y_train = y_train
# Y_test = y_test
Y_train = np.zeros([y_train.shape[0],y_train.shape[1],y_train.shape[2]+1])
Y_train[:,:,0] = w_train
Y_train[:,:,1:] = y_train
Y_test = np.zeros([y_test.shape[0],y_test.shape[1],y_test.shape[2]+1])
Y_test[:,:,0] = w_test
Y_test[:,:,1:] = y_test

# Set the dimensions ordering according to tensorflow consensous
K.set_image_dim_ordering('tf')

# Training setting
model_checkpoint = ModelCheckpoint(file_path + weights_file, verbose=1, save_best_only=True)

# Change learning when loss reaches a plataeu
change_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.005)

# Build model
model = build_model((psize,psize,1))

# Create an image data generator for real time data augmentation
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=0.,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.,  # randomly shift images vertically (fraction of total height)
    zoom_range=0.,
    shear_range=0.,
    horizontal_flip=False,  # randomly flip images
    vertical_flip=False,  # randomly flip images
    fill_mode='constant',
    data_format=K.image_data_format())

# Fit the image generator on the training data
datagen.fit(X_train_norm)

# loss history recorder
history = LossHistory()

print('Fitting model...')
# model.fit(data, label, batch_size=16, epochs=3, verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])
train_history = model.fit_generator(datagen.flow(X_train_norm, Y_train, batch_size=16),
                                    steps_per_epoch=20, epochs=3, verbose=1,
                                    validation_data=(X_test_norm, Y_test),
                                    callbacks=[history, model_checkpoint, change_lr])


# plot the loss function progression during training
loss = train_history.history['loss']
np_loss = np.array(loss)
np.savetxt(file_path+"loss_history.txt", np_loss, delimiter=",")
val_loss = train_history.history['val_loss']
plt.figure()
plt.plot(loss)
plt.plot(val_loss)
plt.legend(['train_loss', 'val_loss'])
plt.xlabel('Iteration #')
plt.ylabel('Loss Function')
plt.title("Loss function progress during training")
plt.show()

# # Save datasets to a matfile to open later in matlab
# mdict = {"mean_test": mean_val_test, "std_test": std_val_test}
# sio.savemat(file_path + meanstd_file, mdict)

""" test (temp) """

# from test import test_model
# file_path = "c_run/"
weights_file = "weights.hdf5"
savename = "recovery_test"
# test_model(X_test_norm, file_path + weights_file, file_path + savename, debug=1)

images = X_test_norm
debug = 1

# (n, r, c, tmp) = images.shape

# Make a prediction and time it
start = time.time()
predicted_density = model.predict(images, batch_size=1)
end = time.time()
print(end - start)

# threshold negative values
predicted_density[predicted_density < 0] = 0

# resulting sum images
WideField = np.squeeze(np.sum(images, axis=0))
Recovery = np.squeeze(np.sum(predicted_density, axis=0))

# Look at the sum image
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True)
ax1.imshow(WideField)
ax1.set_title('Wide Field')
ax2.imshow(Recovery)
ax2.set_title('Sum of Predictions')
f.subplots_adjust(hspace=0)
plt.show()

# Save predictions to a matfile to open later in matlab
mdict = {"Recovery": Recovery}
sio.savemat(file_path + savename + '.mat', mdict)

# save predicted density in each frame for debugging purposes
if debug:
    mdict = {"Predictions": predicted_density}
    sio.savemat(file_path + savename + '_predictions.mat', mdict)

