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
flags.DEFINE_string('dataset', 'mnist', 'path to dataset')
flags.DEFINE_string('save_dir', 'save', 'directory to save the image samples')
flags.DEFINE_boolean('train', True, 'True for training, False for testing')

FLAGS = flags.FLAGS

def main(argv):
    if FLAGS.dataset == 'mnist':
        FLAGS.input_width = FLAGS.input_height = \
            FLAGS.output_width = FLAGS.output_height = 28
        FLAGS.z_dim = 64
        FLAGS.g_dim = 32
        FLAGS.d_dim = 32
        FLAGS.n_channel = 1
        dataset = MnistDataset(code_dim=0, code_init=None)
    elif FLAGS.dataset == 'cifar10':
        FLAGS.input_width = FLAGS.input_height = \
            FLAGS.output_width = FLAGS.output_height = 32
        FLAGS.z_dim = 64
        FLAGS.g_dim = 32
        FLAGS.d_dim = 32
        FLAGS.n_channel = 3
        dataset = Cifar10Dataset('/home/chiba/data/cifar10/cifar-10-batches-py',
            code_dim=0, code_init=None)
    elif FLAGS.dataset == 'celeba':
        FLAGS.input_width = FLAGS.input_height = \
            FLAGS.output_width = FLAGS.output_height = 64
        FLAGS.z_dim = 512
        FLAGS.g_dim = 64
        FLAGS.d_dim = 64
        FLAGS.n_channel = 3
        dataset = CelebADataset('/home/chiba/data/celeba/img_align_celeba/',
            code_dim=0, code_init=None)
    else:
        raise ValueError('Dataset %s is unsupported.'%FLAGS.dataset)

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(FLAGS.__flags)

    if not os.path.exists(FLAGS.save_dir):
        os.makedirs(FLAGS.save_dir)
    FLAGS.image_dir = os.path.join(FLAGS.save_dir, 'images')
    if not os.path.exists(FLAGS.image_dir):
        os.makedirs(FLAGS.image_dir)
    FLAGS.checkpoint_dir = os.path.join(FLAGS.save_dir, 'checkpoints')
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        dcgan = GAN(sess, FLAGS)
        dcgan.build_model()
        if FLAGS.train:
            dcgan.train(dataset=dataset)

if __name__ == '__main__':
    tf.app.run()
