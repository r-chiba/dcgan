from __future__ import print_function
from __future__ import division

import os
import pprint
import numpy as np
import tensorflow as tf

from data import *

flags = tf.app.flags
flags.DEFINE_float('learning_rate', 0.0002, 'learning rate for optimizer')
flags.DEFINE_integer('n_epoch', 30, 'number of epoch')
flags.DEFINE_integer('batch_size', 64, 'batch size')
flags.DEFINE_string('dataset', 'mnist', 'path to dataset [mnist, cifar10, celeba]')
flags.DEFINE_string('save_dir', 'save', 'directory to save image samples')
flags.DEFINE_string('checkpoint_path', '', 'path to trained model')
flags.DEFINE_string('pkl_path', '', 'path to pickled codes')
flags.DEFINE_integer('gpu_list', '0', 'gpu numbers to use')
flags.DEFINE_boolean('train', True, 'True for training, False for testing')

FLAGS = flags.FLAGS

def main(argv):
    if FLAGS.dataset == 'mnist':
        FLAGS.input_width = FLAGS.input_height = \
            FLAGS.output_width = FLAGS.output_height = 28
        FLAGS.z_dim = 100
        FLAGS.g_dim = 32
        FLAGS.d_dim = 32
        FLAGS.n_channel = 1
        dataset = MnistDataset(code_dim=FLAGS.z_dim,
            code_init='pca' if FLAGS.pkl_path=='' else FLAGS.pkl_path)
    elif FLAGS.dataset == 'cifar10':
        FLAGS.input_width = FLAGS.input_height = \
            FLAGS.output_width = FLAGS.output_height = 32
        FLAGS.z_dim = 100
        FLAGS.g_dim = 32
        FLAGS.d_dim = 32
        FLAGS.n_channel = 3
        dataset = Cifar10Dataset('/home/chiba/data/cifar10/cifar-10-batches-py',
            code_dim=FLAGS.z_dim,
            code_init='pca' if FLAGS.pkl_path=='' else FLAGS.pkl_path)
    elif FLAGS.dataset == 'celeba':
        FLAGS.input_width = FLAGS.input_height = \
            FLAGS.output_width = FLAGS.output_height = 64
        FLAGS.z_dim = 100
        FLAGS.g_dim = 64
        FLAGS.d_dim = 64
        FLAGS.n_channel = 3
        dataset = CelebADataset('/home/chiba/data/celeba/img_align_celeba/',
            code_dim=FLAGS.z_dim,
            code_init='pca' if FLAGS.pkl_path=='' else FLAGS.pkl_path)
    else:
        raise ValueError('Dataset %s is unsupported.'%FLAGS.dataset)

    if FLAGS.checkpoint_path == '':
        # pre-training by glo
        from models.glo import GAN
    else:
        # fine-tuning by gmmn
        from models.glo_gmmn import GAN

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
    config.gpu_options.visible_device_list = FLAGS.gpu_list

    with tf.Session(config=config) as sess:
        dcgan = GAN(sess, FLAGS)
        dcgan.build_model()
        if FLAGS.train:
            dcgan.train(dataset=dataset)

if __name__ == '__main__':
    tf.app.run()
