from __future__ import division
from __future__ import print_function
import os
import sys
import cPickle
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import tensorflow as tf
import glob
import cv2

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

class Cifar10BatchGenerator:
    def __init__(self, dataset_path, batch_size):
        self.batch_size = batch_size
        self.image = None
        self.label = None
        for i in xrange(1, 6):
            print('Extracting data_batch_%s...'%i)
            subset_path = os.path.join(dataset_path, 'data_batch_%s'%i)
            if not os.path.exists(subset_path):
                raise ValueError('File %s does not exist.'%subset_path)
            data = self.unpickle(subset_path)
            images = data['data']
            images = images.astype(np.float32) / 255. - .5
            images = images.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
            #images = [x.astype(np.float32) / 255. - .5 for x in data]
            #images = [x.reshape(3, 32, 32).transpose(1, 2, 0)
            #    for x in images]
            if self.image is None:
                self.image = images
            else:
                self.image = np.concatenate((self.image, images), axis=0)
            if self.label is None:
                self.label = self.get_onehot_vectors(data['labels'])
            else:
                self.label = np.concatenate((self.label,
                    self.get_onehot_vectors(data['labels'])), axis=0)

        #self.image = tf.pack(self.image)
        #self.label = tf.pack(self.label)

        self.batch_idx = 0
        self.rand_idx = np.random.permutation(len(self.image))
        print(len(self.image))

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
        #print(type(x))
        #print(x.shape)
        #print(x[0])
        #x = tf.image.per_image_whitening(x)

        return x, t, last_batch

    def unpickle(self, file_path):
        f = open(file_path, 'rb')
        data = cPickle.load(f)
        f.close()
        return data

    def get_onehot_vectors(self, labels):
        x = np.array(labels).reshape(1, -1)
        x = x.transpose()
        encoder = OneHotEncoder(n_values=max(x)+1)
        x = encoder.fit_transform(x).toarray()
        return x

class CelebABatchGenerator:
    def __init__(self, dataset_path, batch_size):
        self.batch_size = batch_size
        self.image = None

        files = glob.glob(os.path.join(dataset_path, '*.jpg'))[:100000]
        images = [cv2.imread(file) for file in files]
        images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]
        images = [cv2.resize(image, (64, 64)) for image in images]
        images = [image.astype(np.float32) / 255. - .5 for image in images]
        #self.image = np.asarray([image.transpose(1, 2, 0) for image in images])
        self.image = np.asarray(images)

        self.batch_idx = 0
        self.rand_idx = np.random.permutation(len(self.image))
        print(len(self.image))

    def __call__(self, color=True):
        idx = self.rand_idx[self.batch_idx*self.batch_size : (self.batch_idx+1)*self.batch_size]

        if (self.batch_idx+2)*self.batch_size > len(self.image)+1:
            last_batch = True
            self.batch_idx = 0
            self.rand_idx = np.random.permutation(len(self.image))
        else:
            last_batch = False
            self.batch_idx += 1

        x = self.image[idx]

        return x, None, last_batch
