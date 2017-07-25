from __future__ import division
from __future__ import print_function
import math
import tensorflow as tf

class BatchNormalization:
    def __init__(self, epsilon=1e-5, decay=0.9, name='batch_norm'):
        self.epsilon = epsilon
        self.decay = decay
        self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x, decay=self.decay, updates_collections=None,
            epsilon=self.epsilon, scale=True, is_training=train, scope=self.name)

def pad_out_size_same(in_size, stride):
    return int(math.ceil(float(in_size) / float(stride)))

def pad_out_size_valid(in_size, filter_size, stride):
    return int(math.ceil(float(in_size - filter_size + 1) / float(stride)))

def conv2d(input_, output_dim, kh=5, kw=5, sth=1, stw=1, sd=0.02, name='conv2d', with_w=False):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [kh, kw, input_.get_shape()[-1], output_dim],
                initializer=tf.truncated_normal_initializer(stddev=sd))
        conv = tf.nn.conv2d(input_, w, strides=[1, sth, stw, 1], padding='SAME')
        bias = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, bias), conv.get_shape())
        if with_w == True:
            return conv, w, bias
        else:
            return conv

def deconv2d(input_, output_shape, kh=5, kw=5, sth=1, stw=1, sd=0.02, name='deconv2d', with_w=False):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [kh, kw, output_shape()[-1], input_.get_shape()[-1]],
                initializer=tf.truncated_normal_initializer(stddev=sd))
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                    strides=[1, sth, stw, 1])
        bias = tf.get_variable('bias', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(deconv, bias), deconv.get_shape())
        if with_w == True:
            return deconv, w, bias
        else:
            return deconv

def lrelu(x, leak=0.2, name='lrelu'):
    return tf.maximum(x, leak*x)

def linear(input_, output_size, vs_name='Linear', sd=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()
    with tf.variable_scope(vs_name):
        w = tf.get_variable('w', [shape[1], output_size], tf.float32,
                tf.random_normal_initializer(stddev=sd))
        b = tf.get_variable('b', [output_size], initializer=tf.constant_initializer(bias_start))
        if with_w == True:
            return tf.matmul(input_, w) + b, w, b
        else:
            return tf.matmul(input_, w) + b
