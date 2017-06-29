from __future__ import print_function
from __future__ import division

import os
import pprint
import numpy as np
import tensorflow as tf

from model import DCGAN

flags = tf.app.flags
flags.DEFINE_integer('epoch', 10, 'epochs to train')
flags.DEFINE_float('learning_rate', 0.0002, 'learning rate for adam')
flags.DEFINE_float('beta1', 0.5, 'momentum term of adam')
flags.DEFINE_integer('batch_size', 64, 'batch size')
flags.DEFINE_integer('input_height', 108, 'height of input image')
flags.DEFINE_integer('input_width', None, 'width of input image')
flags.DEFINE_integer('output_height', 64, 'height of output image')
flags.DEFINE_integer('output_width', None, 'width of output image')
flags.DEFINE_string('data_dir', 'data', 'path to dataset')
flags.DEFINE_string('checkpoint_dir', 'checkpoint', 'directory to save the checkpoints')
flags.DEFINE_string('sample_dir', 'sample', 'directory to save the image samples')
flags.DEFINE_boolean('train', False, 'True for training, False for testing')

FLAGS = flags.FLAGS

def main():
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(flags.FLAGS.__flags)
    
    if FLAGS.input_width is None:
        FLAGS.input_width = FLAGS.input_height
    if FLAGS.output_width is None:
        FLAGS.output_width = FLAGS.output_height

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dirs)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dirs)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        dcgan = DCGAN()
        if FLAGS.train:
            dcgan.train(FLAGS)
        else:
            if not dcgan.load(FLAGS.checkpoint_dir)[0]:
                raise Exception('[!] Train a model first, then run test mode')

if __name__ == '__main__':
    tf.app.run()
