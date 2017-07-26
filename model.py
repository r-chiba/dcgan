from __future__ import print_function
from __future__ import division

import os
import time
import math
import glob
import numpy as np
import tensorflow as tf
import cv2

from utils import *


class Genarator:
    def __init__(self, batch_size, in_d, out_h, out_w, out_c, dim, name='generator', reuse=False):
        self.batch_size = batch_size
        self.dim = dim
        h2, w2 = pad_out_size_same(out_h, 2), pad_out_size_same(out_w, 2)
        h4, w4 = pad_out_size_same(h2, 2), pad_out_size_same(w2, 2)
        h8, w8 = pad_out_size_same(h4, 2), pad_out_size_same(w4, 2)
        h16, w16 = pad_out_size_same(h8, 2), pad_out_size_same(w8, 2)
        self.reshape = (h16, w16)

        with tf.variable_scope(name) as scope:
            if reuse == True:
                scope.reuse_variables()

            self.linear = Linear(in_d, self.dim*8*h16*w16, name='g_h0')

            self.deconv1 = Deconvolution2d(self.dim*8, [self.batch_size, h8, w8, self.dim*4], sth=2, stw=2, name='g_h1')
            self.bn1 = BatchNormalization()

            self.deconv2 = Deconvolution2d(self.dim*4, [self.batch_size, h4, w4, self.dim*2], sth=2, stw=2, name='g_h2')
            self.bn2 = BatchNormalization()
            
            self.deconv3 = Deconvolution2d(self.dim*2, [self.batch_size, h2, w2, self.dim], sth=2, stw=2, name='g_h3')
            self.bn3 = BatchNormalization()

            self.deconv4 = Deconvolution2d(self.dim, [self.batch_size, h, w, out_c], sth=2, stw=2, name='g_h4')
            self.bn4 = BatchNormalization()


    def __call__(self, z, train=True):
        h = self.linear(z)
        h = tf.nn.relu(h)
        h = tf.reshape(h, [self.batch_size, self.reshape[0], self.reshape[1], self.dim*8])

        h = self.deconv1(h)
        h = self.bn1(h, train=train)
        h = tf.nn.relu(h)

        h = self.deconv2(h)
        h = self.bn2(h, train=train)
        h = tf.nn.relu(h)

        h = self.deconv3(h)
        h = self.bn3(h, train=train)
        h = tf.nn.relu(h)

        h = self.deconv4(h)
        h = self.bn4(h, train=train)
        h = tf.nn.relu(h)

        return tf.nn.tanh(h)


class Discriminator:
    def __init__(self, batch_size, in_c, in_h, in_w, dim, name='Discriminator', reuse=False):
        self.batch_size = batch_size
        self.dim = dim
        h2, w2 = pad_out_size_same(in_h, 2), pad_out_size_same(in_w, 2)
        h4, w4 = pad_out_size_same(h2, 2), pad_out_size_same(w2, 2)
        h8, w8 = pad_out_size_same(h4, 2), pad_out_size_same(w4, 2)

        with tf.variable_scope(name) as scope:
            if reuse == True:
                scope.reuse_variables()

            self.conv1 = Convolution2d(in_c, self.dim*2, sth=2, stw=2, name='d_h0')
            self.bn1 = BatchNormalization()

            self.conv2 = Convolution2d(self.dim*2, self.dim*4, sth=2, stw=2, name='d_h1')
            self.bn2 = BatchNormalization()

            self.conv3 = Convolution2d(self.dim*4, self.dim*8, sth=2, stw=2, name='d_h2')
            self.bn3 = BatchNormalization()

            self.linear = Linear(h8*w8*self.dim*8, 1, name='d_h3')


    def __call__(self, x, train=True):
        h = self.conv1(x)
        h = self.bn1(h)
        h = lrelu(h)

        h = self.conv2(x)
        h = self.bn2(h)
        h = lrelu(h)

        h = self.conv3(x)
        h = self.bn3(h)
        h = lrelu(h)

        logit = self.linear(h)

        return logit


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
        self.savedir = flags.save_dir
        self.training = flags.train
        self.n_epoch = flags.n_epoch

        self.Gen = Generator(self.batch_size, self.n_channel, self.output_height, self.output_width, self.n_channel, self.g_dim)
        self.Dis = Discriminator(self.batch_size, self.n_channel, self.output_height, self.output_width, self.d_dim)
        #self.Dis_real = Discriminator(self.batch_size, self.n_channel, self.output_height, self.output_width, self.d_dim)
        #self.Dis_fake = Discriminator(self.batch_size, self.n_channel, self.output_height, self.output_width, self.d_dim, reuse=True)

        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.x_real = tf.placeholder(tf.float32, 
            [self.batch_size, self.input_height, self.input_width, self.n_channel], name='x')
        self.x_fake = self.Gen(self.z)
        self.x_sample = self.Gen(self.z, training=False)
        self.d_real = self.Dis(self.x_real)
        self.d_fake = self.Dis(self.x_fake, reuse=True)

        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.d_real, targets=tf.ones_like(self.d_real)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.d_fake, targets=tf.zeros_like(self.d_fake)))
        self.d_loss = self.d_loss_real + self.d_loss_fake
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.d_fake, targets=tf.ones_like(self.d_fake)))

        self.d_vars = [x for x in tf.trainable_variables() if 'd_' in x.name]
        self.g_vars = [x for x in tf.trainable_variables() if 'g_' in x.name]

        self.d_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1).minimize(
            self.d_loss, var_list=self.d_vars)
        self.g_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1).minimize(
            self.g_loss, var_list=self.g_vars)

        tf.scalar_summary('d_loss_real', self.d_loss_real)
        tf.scalar_summary('d_loss_fake', self.d_loss_fake)
        tf.scalar_summary('d_loss', self.d_loss)
        tf.scalar_summary('g_loss', self.g_loss)

        self.saver = tf.train.Saver()
        self.summary = tf.merge_all_summaries()
        self.writer = tf.train.SummaryWriter(self.savedir, self.sess.graph)

    #def generator(self, z, reuse=False, training=True):
    #    h, w = self.output_height, self.output_width
    #    h2, w2 = pad_out_size_same(h, 2), pad_out_size_same(w, 2)
    #    h4, w4 = pad_out_size_same(h2, 2), pad_out_size_same(w2, 2)
    #    h8, w8 = pad_out_size_same(h4, 2), pad_out_size_same(w4, 2)
    #    h16, w16 = pad_out_size_same(h8, 2), pad_out_size_same(w8, 2)

    #    with tf.variable_scope('generator') as scope:
    #        if reuse == True:
    #            scope.reuse_variables()

    #        hid0, self.h0_w, self.h0_b = linear(z, self.g_dim*8*h16*w16, 'g_h0_lin', with_w=True)
    #        #hid0 = tf.contrib.layers.batch_norm(hid0, decay=0.9,
    #        #    updates_collections=None, epsilon=1e-5, scale=True, is_training=training)
    #        hid0 = tf.nn.relu(hid0)
    #        hid0 = tf.reshape(hid0, [self.batch_size, h16, w16, self.g_dim*8])

    #        hid1, self.h1_w, self.h1_b = deconv2d(hid0, [self.batch_size, h8, w8, self.g_dim*4], 
    #            sth=2, stw=2, name='g_h1', with_w=True)
    #        hid1 = tf.contrib.layers.batch_norm(hid1, decay=0.9,
    #            updates_collections=None, epsilon=1e-5, scale=True, is_training=training)
    #        hid1 = tf.nn.relu(hid1)

    #        hid2, self.h2_w, self.h2_b = deconv2d(hid1, [self.batch_size, h4, w4, self.g_dim*2], 
    #            sth=2, stw=2, name='g_h2', with_w=True)
    #        hid2 = tf.contrib.layers.batch_norm(hid2, decay=0.9,
    #            updates_collections=None, epsilon=1e-5, scale=True, is_training=training)
    #        hid2 = tf.nn.relu(hid2)

    #        hid3, self.h3_w, self.h3_b = deconv2d(hid2, [self.batch_size, h2, w2, self.g_dim], 
    #            sth=2, stw=2, name='g_h3', with_w=True)
    #        hid3 = tf.contrib.layers.batch_norm(hid3, decay=0.9,
    #            updates_collections=None, epsilon=1e-5, scale=True, is_training=training)
    #        hid3 = tf.nn.relu(hid3)

    #        hid4, self.h4_w, self.h4_b = deconv2d(hid3, [self.batch_size, h, w, self.n_channel], 
    #            sth=2, stw=2, name='g_h4', with_w=True)

    #        if reuse == True:
    #            tf.histogram_summary('g_h0_w', self.h0_w)
    #            tf.histogram_summary('g_h0_b', self.h0_b)
    #            tf.histogram_summary('g_h1_w', self.h1_w)
    #            tf.histogram_summary('g_h1_b', self.h1_b)
    #            tf.histogram_summary('g_h2_w', self.h2_w)
    #            tf.histogram_summary('g_h2_b', self.h2_b)
    #            tf.histogram_summary('g_h3_w', self.h3_w)
    #            tf.histogram_summary('g_h3_b', self.h3_b)
    #            tf.histogram_summary('g_h4_w', self.h4_w)
    #            tf.histogram_summary('g_h4_b', self.h4_b)

    #        return tf.nn.tanh(hid4)

    #def discriminator(self, x, reuse=False):
    #    with tf.variable_scope('discriminator') as scope:
    #        if reuse == True:
    #            scope.reuse_variables()

    #        h0, self.h0_w, self.h0_b= conv2d(x, self.d_dim, sth=2, stw=2, 
    #            name='d_h0_conv', with_w=True)
    #        h0 = tf.contrib.layers.batch_norm(h0, decay=0.9,
    #            updates_collections=None, epsilon=1e-5, scale=True, is_training=self.training)
    #        h0 = lrelu(h0)

    #        h1, self.h1_w, self.h1_b= conv2d(h0, self.d_dim*2, sth=2, stw=2, 
    #            name='d_h1_conv', with_w=True)
    #        h1 = tf.contrib.layers.batch_norm(h1, decay=0.9,
    #            updates_collections=None, epsilon=1e-5, scale=True, is_training=self.training)
    #        h1 = lrelu(h1)

    #        h2, self.h2_w, self.h2_b= conv2d(h1, self.d_dim*4, sth=2, stw=2,
    #            name='d_h2_conv', with_w=True)
    #        h2 = tf.contrib.layers.batch_norm(h2, decay=0.9,
    #            updates_collections=None, epsilon=1e-5, scale=True, is_training=self.training)
    #        h2 = lrelu(h2)
    #        
    #        h3, self.h3_w, self.h3_b= conv2d(h2, self.d_dim*8, sth=2, stw=2, 
    #            name='d_h3_conv', with_w=True)
    #        h3 = tf.contrib.layers.batch_norm(h3, decay=0.9,
    #            updates_collections=None, epsilon=1e-5, scale=True, is_training=self.training)
    #        h3 = lrelu(h3)

    #        h4,self.h4_w, self.h4_b = linear(tf.reshape(h3, [self.batch_size, -1]),
    #            1, 'd_h4_lin', with_w=True)

    #        if reuse == False:
    #            tf.histogram_summary('d_h0_w', self.h0_w)
    #            tf.histogram_summary('d_h0_b', self.h0_b)
    #            tf.histogram_summary('d_h1_w', self.h1_w)
    #            tf.histogram_summary('d_h1_b', self.h1_b)
    #            tf.histogram_summary('d_h2_w', self.h2_w)
    #            tf.histogram_summary('d_h2_b', self.h2_b)
    #            tf.histogram_summary('d_h3_w', self.h3_w)
    #            tf.histogram_summary('d_h3_b', self.h3_b)
    #            tf.histogram_summary('d_h4_w', self.h4_w)
    #            tf.histogram_summary('d_h4_b', self.h4_b)

    #        return h4

    #def build_model(self):
    #    self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
    #    self.x_real = tf.placeholder(tf.float32, 
    #        [self.batch_size, self.input_height, self.input_width, self.n_channel], name='x')
    #    self.x_fake = self.generator(self.z)
    #    self.x_sample = self.generator(self.z, reuse=True, training=False)
    #    self.d_real = self.discriminator(self.x_real)
    #    self.d_fake = self.discriminator(self.x_fake, reuse=True)

    #    self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    #        logits=self.d_real, targets=tf.ones_like(self.d_real)))
    #    self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    #        logits=self.d_fake, targets=tf.zeros_like(self.d_fake)))
    #    self.d_loss = self.d_loss_real + self.d_loss_fake
    #    self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    #        logits=self.d_fake, targets=tf.ones_like(self.d_fake)))

    #    self.d_vars = [x for x in tf.trainable_variables() if 'd_' in x.name]
    #    self.g_vars = [x for x in tf.trainable_variables() if 'g_' in x.name]

    #    self.d_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1).minimize(
    #        self.d_loss, var_list=self.d_vars)
    #    self.g_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1).minimize(
    #        self.g_loss, var_list=self.g_vars)

    #    tf.scalar_summary('d_loss_real', self.d_loss_real)
    #    tf.scalar_summary('d_loss_fake', self.d_loss_fake)
    #    tf.scalar_summary('d_loss', self.d_loss)
    #    tf.scalar_summary('g_loss', self.g_loss)

    #    self.saver = tf.train.Saver()
    #    self.summary = tf.merge_all_summaries()
    #    self.writer = tf.train.SummaryWriter(self.savedir, self.sess.graph)

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

        step = 0
        epoch = 1
        start = time.time()
        while epoch <= self.n_epoch:

            batch_images, batch_labels, last_batch = batch_generator()
            batch_z = np.random.uniform(-1., 1., [self.batch_size, self.z_dim]).astype(np.float32)

            _, g_loss = self.sess.run([self.g_optimizer, self.g_loss], 
                feed_dict={self.z: batch_z})
            _, d_loss, x_fake, x_real, summary = self.sess.run(
                [self.d_optimizer, self.d_loss, self.x_fake, self.x_real, self.summary],
                feed_dict={self.z: batch_z, self.x_real: batch_images})

            if step > 0 and step % 10 == 0:
                self.writer.add_summary(summary, step)

            if step % 100 == 0:
                print("epoch %d(%6d): loss(D)=%.4e, loss(G)=%.4e; time/step=%.2f sec" %
                        (epoch, step, d_loss, g_loss, time.time()-start))
                start = time.time()

            if step % 500 == 0:
                z1 = np.random.uniform(-1, 1, [self.batch_size, self.z_dim])
                z2 = np.random.uniform(-1, 1, [self.z_dim])
                z2 = np.expand_dims(z2, axis=0)
                z2 = np.repeat(z2, repeats=self.batch_size, axis=0)

                gimg1 = self.sess.run(self.x_sample, feed_dict={self.z: z1})
                gimg2 = self.sess.run(self.x_sample, feed_dict={self.z: z2})
                cv2.imwrite(os.path.join(self.savedir, "images", "img_%d_real.png" % step), 
                    tile_image(x_real)*255. + 128.)
                cv2.imwrite(os.path.join(self.savedir, "images", "img_%d_fake1.png" % step), 
                    tile_image(gimg1)*255. + 128.)
                cv2.imwrite(os.path.join(self.savedir, "images", "img_%d_fake2.png" % step), 
                    tile_image(gimg2)*255. + 128.)

            step += 1
            if last_batch == True: epoch += 1
