from __future__ import print_function
from __future__ import division

import os
import pprint
import numpy as np
import tensorflow as tf

from model import GAN
from data import *

flags = tf.app.flags
flags.DEFINE_integer('epoch', 10, 'epochs to train')
flags.DEFINE_float('learning_rate', 0.0002, 'learning rate for adam')
flags.DEFINE_float('beta1', 0.5, 'beta1 of adam')
flags.DEFINE_integer('n_epoch', 30, 'number of epoch')
flags.DEFINE_integer('batch_size', 64, 'batch size')
#flags.DEFINE_integer('input_height', 28, 'height of input image')
#flags.DEFINE_integer('input_width', None, 'width of input image')
#flags.DEFINE_integer('output_height', 28, 'height of output image')
#flags.DEFINE_integer('output_width', None, 'width of output image')
flags.DEFINE_string('dataset_name', 'mnist', 'path to dataset')
flags.DEFINE_string('save_dir', 'save', 'directory to save the image samples')
flags.DEFINE_boolean('train', True, 'True for training, False for testing')

FLAGS = flags.FLAGS

def main(argv):
    #if FLAGS.input_width is None:
    #    FLAGS.input_width = FLAGS.input_height
    #if FLAGS.output_width is None:
    #    FLAGS.output_width = FLAGS.output_height
    if FLAGS.dataset_name == 'mnist':
        FLAGS.input_width = FLAGS.input_height = \
            FLAGS.output_width = FLAGS.output_height = 28
        batch = MnistBatchGenerator(batch_size=FLAGS.batch_size)
    elif FLAGS.dataset_name == 'cifar10':
        FLAGS.input_width = FLAGS.input_height = \
            FLAGS.output_width = FLAGS.output_height = 32
        batch = Cifar10BatchGenerator('/home/chiba/data/cifar10/cifar-10-batches-py',
            batch_size=FLAGS.batch_size)
    else:
        raise ValueError('Dataset %s is unsupported.'%FLAGS.dataset_name)

    FLAGS.z_dim = 100
    FLAGS.g_dim = 64
    FLAGS.d_dim = 64
    FLAGS.n_channel = 3

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(FLAGS.__flags)

    if not os.path.exists(FLAGS.save_dir):
        os.makedirs(FLAGS.save_dir)
    image_dir = os.path.join(FLAGS.save_dir, 'images')
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        dcgan = GAN(sess, FLAGS)
        dcgan.build_model()
        if FLAGS.train:
            dcgan.train(batch_generator=batch)

if __name__ == '__main__':
    tf.app.run()
