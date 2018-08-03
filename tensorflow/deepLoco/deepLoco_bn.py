from __future__ import print_function, division, absolute_import, unicode_literals

import os
import shutil
import math
from collections import OrderedDict
import logging

import tensorflow as tf

from deepLoco.nets import deepLoco_decoder

class deepLoco_net(object):

    def __init__(self, img_channels=1, cost="deepLoco_error", cost_kwargs={}, **kwargs):

        # basic variables
        self.img_channels = img_channels

        # placeholders for input x and y
        self.x = tf.placeholder("float32", shape=[None, 64, 64, img_channels])
        self.y = tf.placeholder("float32", shape=[None, 256, 3])
        self.phase = tf.placeholder(tf.bool, name='phase')
        self.keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

        # reused variables
        self.nx = tf.shape(self.x)[1]
        self.ny = tf.shape(self.x)[2]
        self.num_examples = tf.shape(self.x)[0]

        # variables need to be calculated
        self.recons = deepLoco_decoder(self.x)
        self.loss = self._get_cost(cost, cost_kwargs)
        self.valid_loss = self._get_cost(cost, cost_kwargs)
        self.avg_psnr = self._get_measure('abs')
        self.valid_avg_psnr =  self._get_measure('abs')

    def _lap_kernel(self, x, y, sigma):
        '''
        Compute kernel matrix
        :param x: 2/3D position. shape=(batchsize, sample_len, 2)
        :param y: 2/3D position. shape=(batchsize, sample_len, 2)
        :param sigma: often choose to be near the expected localization precision do the system, ie., 20 to 50 nanometers
        '''
        matrix1 = tf.tile(tf.expand_dims(x, 2), [1, 1, 256, 1]) # repeat column
        matrix2 = tf.tile(tf.expand_dims(y, 1), [1, 256, 1, 1])

        return 1/(2*sigma)*tf.exp(-tf.reduce_sum(tf.abs(matrix1-matrix2), 3)/sigma)

    def _batch_dot(self, x1, y, x2):
        x1 = tf.tile(tf.expand_dims(x1,2),[1,1,256])
        z = tf.multiply(x1, y)
        z = tf.reduce_sum(z, 1)
        z = tf.reduce_sum(tf.multiply(z,x2),1)
        return z

    def _deepLoco_absolute_error(self, y_true, y_pred):

        weight_true = y_true[:,:,0]
        position_true = y_true[:,:,1:]
        weight_pred = y_pred[:,:,0]
        position_pred = y_pred[:,:,1:]

        weight_true = tf.tile(tf.expand_dims(weight_true,2),[1,1,2])
        weight_pred = tf.tile(tf.expand_dims(weight_pred,2),[1,1,2])
        error = tf.abs(tf.multiply(weight_true,position_true)-tf.multiply(weight_pred,position_pred))
        return tf.reduce_sum(error)

    def _deepLoco_loss(self, y_pred, y_true):

        weight_true = y_true[:,:,0]
        position_true = y_true[:,:,1:]
        weight_pred = y_pred[:,:,0]
        position_pred = y_pred[:,:,1:]

        loss = tf.Variable(0.0)
        for sigma in range(20,50,10):
            K_tt = self._lap_kernel(position_true,position_true,sigma=sigma)
            K_tp = self._lap_kernel(position_true,position_pred,sigma=sigma)
            K_pp = self._lap_kernel(position_pred,position_pred,sigma=sigma)
            # print(K_tt.shape, K_tp.shape, K_pp.shape)

            s1 = self._batch_dot(weight_true, K_tt, weight_true)
            s2 = self._batch_dot(weight_true, K_tp, weight_pred)
            s3 = self._batch_dot(weight_pred, K_pp, weight_pred)
            print('s1',tf.shape(s1))

            loss = loss + tf.reduce_sum(s1-2*s2+s3)
        # print(loss.shape)
        return loss

    def _get_measure(self, measure):
        # total_pixels = self.nx * self.ny * self.truth_channels
        # dtype       = self.x.dtype
        flat_recons = self.recons
        flat_truths = self.y

        if measure == 'abs':
            result = self._deepLoco_absolute_error(flat_truths, flat_recons)

        # if measure == 'psnr':
        #     # mse are of the same length of the truths
        #     mse = mse_array(flat_recons, flat_truths, total_pixels)
        #     term1 = log(tf.constant(1, dtype), 10.)
        #     term2 = log(mse, 10.)
        #     psnr = tf.scalar_mul(20., term1) - tf.scalar_mul(10., term2)
        #     result = psnr
        #
        # elif measure == 'avg_psnr':
        #     # mse are of the same length of the truths
        #     mse = mse_array(flat_recons, flat_truths, total_pixels)
        #     term1 = log(tf.constant(1, dtype), 10.)
        #     term2 = log(mse, 10.)
        #     psnr = tf.scalar_mul(20., term1) - tf.scalar_mul(10., term2)
        #     avg_psnr = tf.reduce_mean(psnr)
        #     result = avg_psnr

        else:
            raise ValueError("Unknown measure: "%measure)

        return result

    def _get_cost(self, cost_name, cost_kwargs):
        """
        Constructs the cost function.

        """

        flat_recons = self.recons
        flat_truths = self.y
        if cost_name == "mean_squared_error":
            loss = tf.losses.mean_squared_error(flat_recons, flat_truths)
            # the mean_squared_error is equal to the following code
            # se = tf.squared_difference(flat_recons, flat_truths)
            # loss = tf.reduce_mean(se, 1)

        elif cost_name == "deepLoco_error":
            loss = self._deepLoco_loss(flat_recons, flat_truths)
        else:
            raise ValueError("Unknown cost function: "%cost_name)

        return loss


    def predict(self, model_path, x_test, keep_prob, phase):
        """
        Uses the model to create a prediction for the given data

        :param model_path: path to the model checkpoint to restore
        :param x_test: Data to predict on. Shape [n, nx, ny, channels]
        :returns prediction: The unet prediction Shape [n, px, py, labels] (px=nx-self.offset/2)
        """

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            # Initialize variables
            sess.run(init)

            # Restore model weights from previously saved model
            self.restore(sess, model_path)

            prediction = sess.run(self.recons, feed_dict={self.x: x_test})  # set phase to False for every prediction
                            # define operation
        return prediction

    def save(self, sess, model_path):
        """
        Saves the current session to a checkpoint

        :param sess: current session
        :param model_path: path to file system location
        """

        saver = tf.train.Saver()
        save_path = saver.save(sess, model_path)
        return save_path

    def restore(self, sess, model_path):
        """
        Restores a session from a checkpoint

        :param sess: current session instance
        :param model_path: path to file system checkpoint location
        """

        saver = tf.train.Saver()
        saver.restore(sess, model_path+"final/model.cpkt")
        logging.info("Model restored from file: %s" % model_path)


