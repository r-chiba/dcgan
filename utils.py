from __future__ import division
from __future__ import print_function
import math
import tensorflow as tf

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak*x)


def pad_out_size_same(in_size, stride):
    return int(math.ceil(float(in_size) / float(stride)))


def pad_out_size_valid(in_size, filter_size, stride):
    return int(math.ceil(float(in_size - filter_size + 1) / float(stride)))


class Linear:
    def __init__(self, in_d, out_d, sd=0.02, bias_init=0.0, name='linear'):
        with tf.variable_scope(name):
            self.w = tf.get_variable('w', [in_d, out_d], tf.float32,
                tf.truncated_normal_initializer(stddev=sd))
            self.b = tf.get_variable('b', [out_d], initializer=tf.constant_initializer(bias_init))


    def __call__(self, x):
        return tf.matmul(x, self.w) + self.b


    def get_variable(self):
        return self.w, self.b


class Convolution2d:
    def __init__(self, in_c, out_c, kh=5, kw=5, sth=1, stw=1, sd=0.02, name='conv2d'):
        self.sth = sth
        self.stw = stw

        with tf.variable_scope(name):
            self.w = tf.get_variable('w', [kh, kw, in_c, out_c],
                initializer=tf.truncated_normal_initializer(stddev=sd))
            self.b = tf.get_variable('b', [out_c], initializer=tf.constant_initializer(0.0))


    def __call__(self, x):
        ret = tf.nn.conv2d(x, self.w, strides=[1, self.sth, self.stw, 1], padding='SAME')
        ret = tf.reshape(tf.nn.bias_add(ret, self.b), ret.get_shape())
        return ret
     

    def get_variable(self):
        return self.w, self.b


class Deconvolution2d:
    def __init__(self, in_c, out_shape, kh=5, kw=5, sth=1, stw=1, sd=0.02, name='deconv2d'):
        self.out_shape = out_shape
        self.sth = sth
        self.stw = stw

        with tf.variable_scope(name):
            self.w = tf.get_variable('w', [kh, kw, out_shape[-1], in_c],
                    initializer=tf.truncated_normal_initializer(stddev=sd))
            self.b = tf.get_variable('b', [out_shape[-1]], initializer=tf.constant_initializer(0.0))


    def __call__(self, x):
        ret = tf.nn.conv2d_transpose(x, self.w, output_shape=self.out_shape,
                    strides=[1, self.sth, self.stw, 1])
        ret = tf.reshape(tf.nn.bias_add(ret, self.b), ret.get_shape())
        return ret
     

    def get_variable(self):
        return self.w, self.b


class BatchNormalization:
    def __init__(self, epsilon=1e-5, decay=0.9, name='batch_norm'):
        self.epsilon = epsilon
        self.decay = decay
        self.name = name

    def __call__(self, x, train=True, reuse=False):
        return tf.contrib.layers.batch_norm(x, decay=self.decay, updates_collections=None,
            epsilon=self.epsilon, scale=True, is_training=train, reuse=reuse, scope=self.name)


#def linear(input_, output_size, vs_name='Linear', sd=0.02, bias_start=0.0, with_w=False):
#    shape = input_.get_shape().as_list()
#    with tf.variable_scope(vs_name):
#        w = tf.get_variable('w', [shape[1], output_size], tf.float32,
#                tf.random_normal_initializer(stddev=sd))
#        b = tf.get_variable('b', [output_size], initializer=tf.constant_initializer(bias_start))
#        if with_w == True:
#            return tf.matmul(input_, w) + b, w, b
#        else:
#            return tf.matmul(input_, w) + b
#
#def conv2d(input_, output_dim, kh=5, kw=5, sth=1, stw=1, sd=0.02, name='conv2d', with_w=False):
#    with tf.variable_scope(name):
#        w = tf.get_variable('w', [kh, kw, input_.get_shape()[-1], output_dim],
#                initializer=tf.truncated_normal_initializer(stddev=sd))
#        conv = tf.nn.conv2d(input_, w, strides=[1, sth, stw, 1], padding='SAME')
#        bias = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0.0))
#        conv = tf.reshape(tf.nn.bias_add(conv, bias), conv.get_shape())
#        if with_w == True:
#            return conv, w, bias
#        else:
#            return conv
#
#def deconv2d(input_, output_shape, kh=5, kw=5, sth=1, stw=1, sd=0.02, name='deconv2d', with_w=False):
#    with tf.variable_scope(name):
#        w = tf.get_variable('w', [kh, kw, output_shape[-1], input_.get_shape()[-1]],
#                initializer=tf.truncated_normal_initializer(stddev=sd))
#        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
#                    strides=[1, sth, stw, 1])
#        bias = tf.get_variable('bias', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
#        conv = tf.reshape(tf.nn.bias_add(deconv, bias), deconv.get_shape())
#        if with_w == True:
#            return deconv, w, bias
#        else:
#            return deconv

