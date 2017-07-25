from __future__ import print_function
from __future__ import division

import os
import time
import math
import glob
import numpy as np
import tensorflow as tf

from util import *

class GAN:
    #def __init__(self, sess, batch_size, input_height, input_width, n_channel, 
    #            output_height, output_width, z_dim, g_dim, d_dim, dataset_name):
    def __init__(self. sess, flags):
        self.sess = sess
        self.batch_size = flags.batch_size
        self.input_height = flags.input_height
        self.input_width = flags.input_width
        self.n_channel = n_channel
        self.output_height = flags.output_height
        self.output_width = flags.output_width
        self.z_dim = flags.z_dim
        self.g_dim = flags.g_dim
        self.d_dim = flags.d_dim
        self.momentum = flags.momentum
        self.beta1 = flags.beta1
        self.savedir = flags.savedir

        build_model()


    def generator(self, z, training=True):
        h, w = self.output_height, self.output_width
        h2, w2 = pad_out_size_same(h, 2), pad_out_size_same(w, 2)
        h4, w4 = pad_out_size_same(h2, 2), pad_out_size_same(w2, 2)
        h8, w8 = pad_out_size_same(h4, 2), pad_out_size_same(w4, 2)
        h16, w16 = pad_out_size_same(h8, 2), pad_out_size_same(w8, 2)

        with tf.variable_scope('generator'):
            hid0, self.h0_w, self.h0_b = linear(z, self.g_dim*8*h16*w16, 'g_h0_lin', with_w=True)
            hid0 = tf.reshape(hid0, [-1, h16, w16, 8])
            hid0 = tf.nn.relu(self.bn0(hid0))

            hid1, self.h1_w, self.h1_b = deconv2d(hid0, [self.batch_size, h8, w8, self.g_dim*4], 
                name='g_h1', with_w=True)
            hid1 = tf.contrib.layers.batch_norm(hid1, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True, is_training=training)
            hid1 = tf.nn.relu(hid1)

            hid2, self.h2_w, self.h2_b = deconv2d(hid1, [self.batch_size, h4, w4, self.g_dim*2], 
                name='g_h2', with_w=True)
            hid2 = tf.contrib.layers.batch_norm(hid2, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True, is_training=training)
            hid2 = tf.nn.relu(hid2)

            hid3, self.h3_w, self.h3_b = deconv2d(hid2, [self.batch_size, h2, w2, self.g_dim], 
                name='g_h3', with_w=True)
            hid3 = tf.contrib.layers.batch_norm(hid3, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True, is_training=training)
            hid3 = tf.nn.relu(hid3)

            hid4, self.h4_w, self.h4_b = deconv2d(hid3, [self.batch_size, h, w, self.n_channel], 
                name='g_h4', with_w=True)

            tf.summary.histgram('g_h0_w', self.h0_w)
            tf.summary.histgram('g_h0_b', self.h0_b)
            tf.summary.histgram('g_h1_w', self.h1_w)
            tf.summary.histgram('g_h1_b', self.h1_b)
            tf.summary.histgram('g_h2_w', self.h2_w)
            tf.summary.histgram('g_h2_b', self.h2_b)
            tf.summary.histgram('g_h3_w', self.h3_w)
            tf.summary.histgram('g_h3_b', self.h3_b)
            tf.summary.histgram('g_h4_w', self.h4_w)
            tf.summary.histgram('g_h4_b', self.h4_b)

            return tf.nn.tanh(hid4)

    def discriminator(self, x, reuse=False):
        with tf.variable_scope('discriminator') as scope:
            if reuse == True:
                scope.reuse_variables()

            h0, self.h0_w, self.h0_b= conv2d(x, self.f_dim, name='d_h0_conv', with_w=True)
            h0 = tf.contrib.layers.batch_norm(h0, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True, is_training=training)
            h0 = lrelu(h0)

            h1, self.h1_w, self.h1_b= conv2d(h0, self.f_dim*2, name='d_h1_conv', with_w=True)
            h1 = tf.contrib.layers.batch_norm(h1, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True, is_training=training)
            h1 = lrelu(h1)

            h2, self.h2_w, self.h2_b= conv2d(h1, self.f_dim*4, name='d_h2_conv', with_w=True)
            h2 = tf.contrib.layers.batch_norm(h2, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True, is_training=training)
            h2 = lrelu(h2)
            
            h3, self.h3_w, self.h3_b= conv2d(h2, self.f_dim*8, name='d_h3_conv', with_w=True)
            h3 = tf.contrib.layers.batch_norm(h3, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True, is_training=training)
            h3 = lrelu(h3)

            h4,self.h4_w, self.h4_b = linear(tf.reshape(h3, [self.batch_size, -1]),
                1, 'd_h4, lin', with_w=True)

            if reuse == False:
                tf.summary.histgram('d_h0_w', self.h0_w)
                tf.summary.histgram('d_h0_b', self.h0_b)
                tf.summary.histgram('d_h1_w', self.h1_w)
                tf.summary.histgram('d_h1_b', self.h1_b)
                tf.summary.histgram('d_h2_w', self.h2_w)
                tf.summary.histgram('d_h2_b', self.h2_b)
                tf.summary.histgram('d_h3_w', self.h3_w)
                tf.summary.histgram('d_h3_b', self.h3_b)
                tf.summary.histgram('d_h4_w', self.h4_w)
                tf.summary.histgram('d_h4_b', self.h4_b)

            return h4

    def build_model(self):
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.x_real = tf.placeholder(tf.float32, 
            [self.batch_size, self.input_height, self.input_width, self.n_channel], name='x')
        self.x_fake = self.generator(self.z)
        self.x_sample = self.generator(self.z, reuse=True, training=False)
        self.d_real = self.discriminator(self.x_real)
        self.d_fake = self.discriminator(self.x_fake)

        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.d_real, label=tf.ones_like(self.d_real)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.d_fake, label=tf.zeros_like(self.d_fake)))
        self.d_loss = self.d_loss_real + self.d_loss_fake
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.d_fake, labels=tf.ones_like(self.d_fake)))

        self.d_vars = [x for x in tf.trainable_variables() if 'd_' in x.name]
        self.g_vars = [x for x in tf.trainable_variables() if 'g_' in x.name]

        self.d_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1).minimize(
            self.d_loss, var_list=self.d_vars)
        self.g_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1).minimize(
            self.g_loss. var_list=self.g_vars)

        tf.summary_scalar('d_loss_real', self.d_loss_real)
        tf.summary_scalar('d_loss_fake', self.d_loss_fake)
        tf.summary_scalar('d_loss', self.d_loss)
        tf.summary_scalar('g_loss', self.g_loss)

        self.saver = tf.train.Saver()
        self.summary = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.savefig, self.sess.graph)

    def train(self, batch_generator):
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

        step = -1
        start = time.time()
        while True:
            step += 1

            batch_images, batch_labels = batch_generator()
            batch_z = np.random.uniform(-1., 1., [self.batch_size, self.z_dim]).astype(np.float32)

            _, g_loss = self.sess.run([self.g_optimizer, self.g_loss], 
                feed_dict={self.z: batch_z})
            _, d_loss, y_fake, y_real, summary = self.sess.run([self.d_optimizer, self.d_loss, self.y])

            if step > 0 and step % 10 == 0:
                self.writer.add_summary(summry, step)

            if step % 100 == 0:
                print("%6d: loss(D)=%.4e, loss(G)=%.4e; time/step=%.2f sec" %
                        (step, d_loss, g_loss, time.time()-start))
                start = time.time()

                z1 = np.random.uniform(-1, 1, [self.batch_size, self.z_dim])
                z2 = np.random.uniform(-1, 1, [self.z_dim])
                z2 = np.expand_dims(z2, axis=0)
                z2 = np.repeat(z2, z2, repeats=self.batch_size, axis=0)

                gimg1 = self.sess.run(self.x_sample, feed_dict={self.z: z1})
                gimg2 = self.sess.run(self.x_sample, feed_dict={self.z: z2})
                cv2.imwrite(os.path.join(self.savedir, "images", "img_%d_real.png" % step), 
                    tile_image(x_real)*255. + 128.)
                cv2.imwrite(os.path.join(self.savedir, "images", "img_%d_fake1.png" % step), 
                    tile_image(gimg1)*255. + 128.)
                cv2.imwrite(os.path.join(self.savedir, "images", "img_%d_fake2.png" % step), 
                    tile_image(gimg2)*255. + 128.)
