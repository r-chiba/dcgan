from __future__ import print_function
from __future__ import division

import os
import sys
import time
import math
import glob
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
import cv2

from utils import *

class GAN:
    def __init__(self, sess, flags):
        self.sess = sess
        self.batch_size = flags.batch_size
        self.input_height = flags.input_height
        self.input_width = flags.input_width
        self.n_channel = flags.n_channel
        self.output_height = flags.output_height
        self.output_width = flags.output_width
        self.z_dim = flags.z_dim
        self.g_dim = flags.g_dim
        self.learning_rate = flags.learning_rate
        self.save_dir = flags.save_dir
        self.image_dir = flags.image_dir
        self.checkpoint_dir = flags.checkpoint_dir
        self.training = flags.train
        self.n_epoch = flags.n_epoch
        self.lr_z = 10.

    def generator(self, z, reuse=False, training=True):
        ht, wh = self.output_height, self.output_width
        ht2, wh2 = pad_out_size_same(ht, 2), pad_out_size_same(wh, 2)
        ht4, wh4 = pad_out_size_same(ht2, 2), pad_out_size_same(wh2, 2)
        ht8, wh8 = pad_out_size_same(ht4, 2), pad_out_size_same(wh4, 2)
        ht16, wh16 = pad_out_size_same(ht8, 2), pad_out_size_same(wh8, 2)

        with tf.variable_scope('generator') as scope:
            if reuse == True:
                scope.reuse_variables()

            h0 = linear(z, self.g_dim*8*ht16*wh16, 'h0', with_w=False)
            #h0 = tf.contrib.layers.batch_norm(h0, decay=0.9,
            #    updates_collections=None, epsilon=1e-5, scale=True, is_training=training)
            h0 = tf.nn.relu(h0)
            h0 = tf.reshape(h0, [self.batch_size, ht16, wh16, self.g_dim*8])

            h1 = deconv2d(h0, [self.batch_size, ht8, wh8, self.g_dim*4], 
                sth=2, stw=2, name='h1', with_w=False)
            h1 = tf.contrib.layers.batch_norm(h1, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True, is_training=training)
            h1 = tf.nn.relu(h1)

            h2 = deconv2d(h1, [self.batch_size, ht4, wh4, self.g_dim*2], 
                sth=2, stw=2, name='h2', with_w=False)
            h2 = tf.contrib.layers.batch_norm(h2, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True, is_training=training)
            h2 = tf.nn.relu(h2)

            h3 = deconv2d(h2, [self.batch_size, ht2, wh2, self.g_dim], 
                sth=2, stw=2, name='h3', with_w=False)
            h3 = tf.contrib.layers.batch_norm(h3, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True, is_training=training)
            h3 = tf.nn.relu(h3)

            h4 = deconv2d(h3, [self.batch_size, ht, wh, self.n_channel], 
                sth=2, stw=2, name='h4', with_w=False)
            h4 = tf.nn.sigmoid(h4)

            return h4

    def build_model(self):
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.x_real = tf.placeholder(tf.float32, 
            [self.batch_size, self.input_height, self.input_width, self.n_channel], name='x')
        self.x_fake = self.generator(self.z)
        self.x_sample = self.generator(self.z, reuse=True, training=False)

        #self.x_fake_reshape = tf.reshape(self.x_fake, [self.batch_size, -1])
        #self.x_sample_reshape = tf.reshape(self.x_sample, [self.batch_size, -1])

        #self.x_dim = self.output_height * self.output_width * self.n_channel
        #m = self.z_dim * 5 * math.log(float(self.x_dim))
        #self.measurement_matrix = tf.random_normal([self.x_dim, self.x_dim], mean=0.0, stddev=1/m)

        #self.x_fake_obs = tf.matmul(self.x_fake_reshape, self.measurement_matrix)
        #self.x_sample_obs = tf.matmul(self.x_sample_reshape, self.measurement_matrix)

        #self.x_fake_obs = tf.reshape(self.x_fake_obs, [self.batch_size, self.output_height, self.output_width, self.n_channel])
        #self.x_sample_obs = tf.reshape(self.x_sample_obs, [self.batch_size, self.output_height, self.output_width, self.n_channel])


        self.loss = tf.reduce_mean(tf.abs(self.x_real - self.x_fake))
        self.loss += laplacian_pyramid_loss(self.x_real, self.x_fake, 4)
        #self.g_loss = tf.reduce_mean(tf.abs(self.x_real - self.x_fake_obs))
        #self.g_loss += laplacian_pyramid_loss(self.x_real, self.x_fake_obs, 4)

        #self.z_loss = tf.reduce_mean(tf.abs(self.x_real - self.x_fake_obs))
        #self.z_loss += laplacian_pyramid_loss(self.x_real, self.x_fake_obs, 4)
        #self.z_loss += 0.002 * tf.nn.l2_loss(self.z)

        self.z_gradient = tf.gradients(self.loss, self.z)
        #self.z_gradient = tf.gradients(self.z_loss, self.z)

        #self.optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5).minimize(
        #    self.loss)
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
        #self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.g_loss)

        self.saver = tf.train.Saver()

    def train(self, dataset):
        def tile_image(imgs):
            d = int(math.sqrt(imgs.shape[0]-1))+1
            h = imgs[0].shape[0]
            w = imgs[0].shape[1]
            r = np.zeros((h*d, w*d, 3), dtype=np.float32)
            for idx, img in enumerate(imgs):
                idx_y = int(idx/d)
                idx_x = idx-idx_y*d
                r[idx_y*h:(idx_y+1)*h, idx_x*w:(idx_x+1)*w, :] = img
            return r

        self.sess.run(tf.initialize_all_variables())

        step = 1
        epoch = 1
        start = time.time()
        while epoch <= self.n_epoch:
            batch_images, _, batch_codes, batch_idxs, last_batch =\
                dataset.train_batch(self.batch_size, with_idx=True)

            _, loss, x_real, x_fake = self.sess.run(
                [self.optimizer, self.loss, self.x_real, self.x_fake],
                feed_dict={self.x_real: batch_images, self.z: batch_codes})

            z, z_grad = self.sess.run([self.z, self.z_gradient],
                feed_dict={self.x_real: batch_images, self.z: batch_codes})
            z_update = z - self.lr_z * z_grad[0]
            z_update = z_update / np.linalg.norm(z_update, axis=1, keepdims=True)
            dataset.codes[batch_idxs] = z_update

            if step % 100 == 0:
                elapsed = time.time() - start
                print("epoch %3d(%6d): loss=%.4e; time/step=%.2f sec" %
                        (epoch, step, loss, elapsed if step == 0 else elapsed/100))
                start = time.time()

            if step % 100 == 0:
                img_real = tile_image(x_real) * 255.
                img_real = cv2.cvtColor(img_real, cv2.COLOR_RGB2BGR)

                img_fake = tile_image(x_fake) * 255.
                img_fake = cv2.cvtColor(img_fake, cv2.COLOR_RGB2BGR)

                z = np.random.randn(self.batch_size, self.z_dim)
                z = z / np.linalg.norm(z, axis=1, keepdims=True)
                img_sample = self.sess.run(self.x_sample, feed_dict={self.z: z})
                #img_sample = self.sess.run(self.x_sample_obs, feed_dict={self.z: z})
                img_sample = tile_image(img_sample) * 255.
                img_sample = cv2.cvtColor(img_sample, cv2.COLOR_RGB2BGR)

                if step == 100:
                    self.z_fix = np.random.randn(self.batch_size, self.z_dim)
                    self.z_fix = self.z_fix / np.linalg.norm(self.z_fix, axis=1, keepdims=True)
                img_sample_fix = self.sess.run(self.x_sample, feed_dict={self.z: self.z_fix})
                img_sample_fix = img_sample_fix[:100]
                img_sample_fix = tile_image(img_sample_fix) * 255.
                img_sample_fix = cv2.cvtColor(img_sample_fix, cv2.COLOR_RGB2BGR)

                cv2.imwrite(os.path.join(self.image_dir, "img_%d_real.png" % step), 
                    img_real)
                cv2.imwrite(os.path.join(self.image_dir, "img_%d_fake.png" % step), 
                    img_fake)
                cv2.imwrite(os.path.join(self.image_dir, "img_%d_sample.png" % step), 
                    img_sample)
                cv2.imwrite(os.path.join(self.image_dir, "img_%d_sample_fix.png" % step), 
                    img_sample_fix)

            step += 1
            if step > 0 and step % 1000 == 0:
                self.lr_z *= 0.95

            if last_batch:
                if epoch > 0 and (epoch % 10 == 0 or epoch == self.n_epoch):
                    ckpt_name = 'glo_epoch-%d.ckpt'%epoch
                    self.saver.save(self.sess, os.path.join(self.checkpoint_dir, ckpt_name))
                    print('save trained model to ' + ckpt_name)
                epoch += 1

            sys.stdout.flush()
            sys.stderr.flush()

        dataset.save_codes(os.path.join(self.save_dir, 'codes.pkl'))
