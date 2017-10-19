from __future__ import print_function
from __future__ import division

import os
import sys
import time
import math
import glob
import numpy as np
import tensorflow as tf
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
        self.d_dim = flags.d_dim
        self.learning_rate = flags.learning_rate
        self.beta1 = flags.beta1
        self.save_dir = flags.save_dir
        self.image_dir = flags.image_dir
        self.checkpoint_dir = flags.checkpoint_dir
        self.training = flags.train
        self.n_epoch = flags.n_epoch

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

    def discriminator(self, x, reuse=False):
        with tf.variable_scope('discriminator') as scope:
            if reuse == True:
                scope.reuse_variables()

            h0 = conv2d(x, self.d_dim, sth=2, stw=2, name='h0', with_w=False)
            h0 = tf.contrib.layers.batch_norm(h0, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True, is_training=self.training)
            h0 = lrelu(h0)

            h1 = conv2d(h0, self.d_dim*2, sth=2, stw=2, name='h1', with_w=False)
            h1 = tf.contrib.layers.batch_norm(h1, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True, is_training=self.training)
            h1 = lrelu(h1)

            h2 = conv2d(h1, self.d_dim*4, sth=2, stw=2, name='h2', with_w=False)
            h2 = tf.contrib.layers.batch_norm(h2, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True, is_training=self.training)
            h2 = lrelu(h2)
            
            h3 = conv2d(h2, self.d_dim*8, sth=2, stw=2, name='h3', with_w=False)
            h3 = tf.contrib.layers.batch_norm(h3, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True, is_training=self.training)
            h3 = lrelu(h3)

            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'h4', with_w=False)

            # return logits
            return h4

    def build_model(self):
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.x_real = tf.placeholder(tf.float32, 
            [self.batch_size, self.input_height, self.input_width, self.n_channel], name='x')
        self.x_fake = self.generator(self.z)
        self.x_sample = self.generator(self.z, reuse=True, training=False)
        self.d_real = self.discriminator(self.x_real)
        self.d_fake = self.discriminator(self.x_fake, reuse=True)

        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.d_real, targets=tf.ones_like(self.d_real)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.d_fake, targets=tf.zeros_like(self.d_fake)))
        self.d_loss = self.d_loss_real + self.d_loss_fake
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.d_fake, targets=tf.ones_like(self.d_fake)))

        self.d_vars = [x for x in tf.trainable_variables() if 'discriminator' in x.name]
        self.g_vars = [x for x in tf.trainable_variables() if 'generator' in x.name]

        self.d_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1).minimize(
            self.d_loss, var_list=self.d_vars)
        self.g_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1).minimize(
            self.g_loss, var_list=self.g_vars)

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

            batch_images, batch_labels, last_batch = dataset.train_batch(self.batch_size)
            batch_z = np.random.uniform(-1., 1., [self.batch_size, self.z_dim]).astype(np.float32)

            _, g_loss = self.sess.run([self.g_optimizer, self.g_loss], 
                feed_dict={self.z: batch_z})
            _, d_loss, x_fake, x_real = self.sess.run(
                [self.d_optimizer, self.d_loss, self.x_fake, self.x_real],
                feed_dict={self.z: batch_z, self.x_real: batch_images})

            if step % 100 == 0:
                elapsed = time.time() - start
                print("epoch %3d(%6d): loss(D)=%.4e, loss(G)=%.4e; time/step=%.2f sec" %
                        (epoch, step, d_loss, g_loss, elapsed if step == 0 else elapsed/100))
                start = time.time()

            if step % 500 == 0:
                img_real = tile_image(x_real) * 255.
                img_real = cv2.cvtColor(img_real, cv2.COLOR_RGB2BGR)

                z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim])
                img_fake = self.sess.run(self.x_sample, feed_dict={self.z: z})
                img_fake = tile_image(x_fake) * 255.
                img_fake = cv2.cvtColor(img_fake, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(self.image_dir, "img_%d_real.png" % step), 
                    img_real)
                cv2.imwrite(os.path.join(self.image_dir, "img_%d_fake.png" % step), 
                    img_fake)

            step += 1
            if last_batch:
                if epoch > 0 and epoch % 10 == 0:
                    ckpt_name = 'dcgan_epoch-%d.ckpt'%epoch
                    self.saver.save(self.sess, os.path.join(self.checkpoint_dir, ckpt_name))
                    print('save trained model to ' + ckpt_name)
                epoch += 1

            sys.stdout.flush()
            sys.stderr.flush()

