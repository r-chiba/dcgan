from __future__ import division
from __future__ import print_function
import os
import sys
import numpy as np
import tensorflow as tf

class MnistBatchGenerator:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets('MNIST_data/', one_hot=False)
        self.image = mnist.train.images
        self.image = np.reshape(self.image, [len(self.image), 28, 28])
        self.label = mnist.train.labels
        self.batch_idx = 0
        self.rand_idx = np.random.permutation(len(self.image))

    def __call__(self, color=True):
        idx = self.rand_idx[self.batch_idx*self.batch_size : (self.batch_idx+1)*self.batch_size]

        if (self.batch_idx+2)*self.batch_size > len(self.image)+1:
            last_batch = True
            self.batch_idx = 0
            self.rand_idx = np.random.permutation(len(self.image))
        else:
            last_batch = False
            self.batch_idx += 1

        x, t = self.image[idx], self.label[idx]
        x = (x - 0.5) / 1.0
        if color == True:
            x = np.expand_dims(x, axis=3)
            x = np.tile(x, (1, 1, 3))
        return x, t, last_batch
