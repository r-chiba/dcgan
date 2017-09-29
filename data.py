from __future__ import division
from __future__ import print_function
import os
import sys
import cPickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
import numpy as np
import tensorflow as tf
import glob
import cv2

def perform_pca(data, dim):
    data = np.reshape(data, (data.shape[0], np.prod(data.shape[1:])))
    pca = PCA(n_components=dim)
    return pca.fit_transform(data)

class MnistDataset:
    def __init__(self, code_dim, code_init, color=False):
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
        self.images = mnist.train.images
        self.images = np.reshape(self.images, [len(self.images), 28, 28, 1])
        self.labels = mnist.train.labels
        if code_init == None:
            self.codes = None
        elif code_init == 'gaussian':
            self.codes = np.random.randn(len(self.images), code_dim)
        elif code_init == "pca":
            print('processing pca...')
            self.codes = perform_pca(self.images, code_dim)
            print('done.')
        if self.codes is not None:
            self.codes = np.asarray([code / np.linalg.norm(code, 2) for code in self.codes])

        if color == True:
            #self.images = np.asarray([np.tile(np.expand_dims(image, axis=3),
            #    (1, 1, 3)) for image in self.images])
            self.images = np.asarray([np.tile(image(1, 1, 3)) for image in self.images])

        self.train_size = int(0.8*len(self.images))
        self.train_images = self.images[:self.train_size]
        self.test_images = self.images[self.train_size:]
        self.train_labels = self.labels[:self.train_size]
        self.test_labels = self.labels[self.train_size:]
        if self.codes is not None:
            self.train_codes = self.codes[:self.train_size]
            self.test_codes = self.codes[self.train_size:]

        self.read_idx = -1
        self.rand_idx = np.random.permutation(self.train_size)

    def train_batch(self, size, with_idx=False):
        idx = self.rand_idx[self.read_idx+1 : self.read_idx+1 + size]

        if self.read_idx+1 + size*2 > self.train_size+1:
            last_batch = True
            self.read_idx = -1
            self.rand_idx = np.random.permutation(self.train_size)
        else:
            last_batch = False
            self.read_idx += size

        if with_idx:
            if self.codes is not None:
                return self.train_images[idx], self.train_labels[idx],\
                    self.train_codes[idx], idx, last_batch
            else:
                return self.train_images[idx], self.train_labels[idx],\
                    idx, last_batch
        else:
            if self.codes is not None:
                return self.train_images[idx], self.train_labels[idx],\
                    self.train_codes[idx], last_batch
            else:
                return self.train_images[idx], self.train_labels[idx],\
                    last_batch

    def test_batch(self, size, with_idx=False):
        idx = np.random.permutation(len(self.images)-self.train_size)[:size]
        if with_idx:
            if self.codes is not None:
                return self.test_images[idx], self.test_labels[idx],\
                    self.test_codes[idx], idx
            else:
                return self.test_images[idx], self.test_labels[idx], idx
        else:
            if self.codes is not None:
                return self.test_images[idx], self.test_labels[idx],\
                    self.test_codes[idx]
            else:
                return self.test_images[idx], self.test_labels[idx]

    def _unpickle(self, file_path):
        f = open(file_path, 'rb')
        data = cPickle.load(f)
        f.close()
        return data

    def _get_onehot_vectors(self, labels):
        x = np.array(labels).reshape(1, -1)
        x = x.transpose()
        encoder = OneHotEncoder(n_values=max(x)+1)
        x = encoder.fit_transform(x).toarray()
        return x

class Cifar10Dataset:
    def __init__(self, dataset_path, code_dim, code_init):
        self.images = None
        self.labels = None
        self.names = None
        self.codes = None
        self.code_dim = code_dim
        for i in xrange(1, 6):
            print('Extracting data_batch_%s...'%i)
            subset_path = os.path.join(dataset_path, 'data_batch_%s'%i)
            if not os.path.exists(subset_path):
                raise ValueError('File %s does not exist.'%subset_path)
            data = self._unpickle(subset_path)
            images = data['data']
            images = images.astype(np.float32) / 255. - .5
            images = images.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
            if self.images is None:
                self.images = images
            else:
                self.images = np.concatenate((self.images, images), axis=0)
            if self.labels is None:
                self.labels = self._get_onehot_vectors(data['labels'])
            else:
                self.labels = np.concatenate((self.labels,
                    self._get_onehot_vectors(data['labels'])), axis=0)
            if self.names is None:
                self.names = data['filenames']
            else:
                self.names = np.concatenate((self.names,
                    data['filenames']), axis=0)
        print('Extracting test_batch...')
        subset_path = os.path.join(dataset_path, 'test_batch')
        if not os.path.exists(subset_path):
            raise ValueError('File %s does not exist.'%subset_path)
        data = self._unpickle(subset_path)
        images = data['data']
        images = images.astype(np.float32) / 255. - .5
        images = images.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
        self.images = np.concatenate((self.images, images), axis=0)
        self.labels = np.concatenate((self.labels,
                self._get_onehot_vectors(data['labels'])), axis=0)
        self.names = np.concatenate((self.names,
                data['filenames']), axis=0)
        
        if code_init == 'gaussian':
            self.codes = np.random.randn(len(self.images), code_dim)
        elif code_init == "pca":
            print('processing pca...')
            self.codes = perform_pca(self.images, code_dim)
            print('done.')
        
        if self.codes is not None:
            self.codes = np.asarray([code / np.linalg.norm(code, 2) for code in self.codes])

        self.train_size = 50000
        self.train_images = self.images[:self.train_size]
        self.test_images = self.images[self.train_size:]
        self.train_labels = self.labels[:self.train_size]
        self.test_labels = self.labels[self.train_size:]
        if self.codes is not None:
            self.train_codes = self.codes[:self.train_size]
            self.test_codes = self.codes[self.train_size:]

        self.read_idx = -1
        self.rand_idx = np.random.permutation(self.train_size)

    def train_batch(self, size, with_idx=False):
        idx = self.rand_idx[self.read_idx+1 : self.read_idx+1 + size]

        if self.read_idx+1 + size*2 > self.train_size+1:
            last_batch = True
            self.read_idx = -1
            self.rand_idx = np.random.permutation(self.train_size)
        else:
            last_batch = False
            self.read_idx += size

        if with_idx:
            if self.codes is not None:
                return self.train_images[idx], self.train_labels[idx],\
                    self.train_codes[idx], idx, last_batch
            else:
                return self.train_images[idx], self.train_labels[idx],\
                    idx, last_batch
        else:
            if self.codes is not None:
                return self.train_images[idx], self.train_labels[idx],\
                    self.train_codes[idx], last_batch
            else:
                return self.train_images[idx], self.train_labels[idx],\
                    last_batch

    def test_batch(self, size, with_idx=False):
        idx = np.random.permutation(len(self.images)-self.train_size)[:size]
        if with_idx:
            if self.codes is not None:
                return self.test_images[idx], self.test_labels[idx],\
                    self.test_codes[idx], idx
            else:
                return self.test_images[idx], self.test_labels[idx], idx
        else:
            if self.codes is not None:
                return self.test_images[idx], self.test_labels[idx],\
                    self.test_codes[idx]
            else:
                return self.test_images[idx], self.test_labels[idx]

    def _unpickle(self, file_path):
        f = open(file_path, 'rb')
        data = cPickle.load(f)
        f.close()
        return data

    def _get_onehot_vectors(self, labels):
        x = np.array(labels).reshape(1, -1)
        x = x.transpose()
        encoder = OneHotEncoder(n_values=max(x)+1)
        x = encoder.fit_transform(x).toarray()
        return x

class CelebADataset:
    def __init__(self, dataset_path, code_dim, code_init):
        self.images = None
        self.codes = None

        #if os.path.exists('data_celeba/'):
        #    pkls = glob.glob(os.path.join('data_celeba', '*.pkl'))
        #    for pkl in pkls:
        #        with open(pkl, 'rb') as f:
        #            data = cPickle.load(f)
        #        if self.images is None:
        #            self.images = data
        #        else:
        #            self.images = np.concatenate((self.images, data), axis=0)
        #    print(len(self.images))

        if self.images is None:
            print('load images...')
            files = glob.glob(os.path.join(dataset_path, '*.jpg'))
            images = [cv2.imread(file) for file in files]
            images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]
            images = [cv2.resize(image, (64, 64)) for image in images]
            images = [image.astype(np.float32) / 255. - .5 for image in images]
            self.images = np.asarray(images)
            print('done.')

        self.labels = np.zeros([len(self.images), 1])

        if code_init == 'gaussian':
            self.codes = np.random.randn(len(self.images), code_dim)
        elif code_init == "pca":
            print('processing pca...')
            self.codes = perform_pca(self.images, code_dim)
            print('done.')
        
        if self.codes is not None:
            self.codes = np.asarray([code / np.linalg.norm(code, 2) for code in self.codes])

        self.train_size = int(0.8*len(self.images))
        self.train_images = self.images[:self.train_size]
        self.test_images = self.images[self.train_size:]
        self.train_labels = self.labels[:self.train_size]
        self.test_labels = self.labels[self.train_size:]
        if self.codes is not None:
            self.train_codes = self.codes[:self.train_size]
            self.test_codes = self.codes[self.train_size:]

        self.read_idx = -1
        self.rand_idx = np.random.permutation(self.train_size)

    def train_batch(self, size, with_idx=False):
        idx = self.rand_idx[self.read_idx+1 : self.read_idx+1 + size]

        if self.read_idx+1 + size*2 > self.train_size+1:
            last_batch = True
            self.read_idx = -1
            self.rand_idx = np.random.permutation(self.train_size)
        else:
            last_batch = False
            self.read_idx += size

        if with_idx:
            if self.codes is not None:
                return self.train_images[idx], self.train_labels[idx],\
                    self.train_codes[idx], idx, last_batch
            else:
                return self.train_images[idx], self.train_labels[idx],\
                    idx, last_batch
        else:
            if self.codes is not None:
                return self.train_images[idx], self.train_labels[idx],\
                    self.train_codes[idx], last_batch
            else:
                return self.train_images[idx], self.train_labels[idx],\
                    last_batch

    def test_batch(self, size, with_idx=False):
        idx = np.random.permutation(len(self.images)-self.train_size)[:size]
        if with_idx:
            if self.codes is not None:
                return self.test_images[idx], self.test_labels[idx],\
                    self.test_codes[idx], idx
            else:
                return self.test_images[idx], self.test_labels[idx], idx
        else:
            if self.codes is not None:
                return self.test_images[idx], self.test_labels[idx],\
                    self.test_codes[idx]
            else:
                return self.test_images[idx], self.test_labels[idx]

#class MnistBatchGenerator:
#    def __init__(self, batch_size):
#        self.batch_size = batch_size
#        from tensorflow.examples.tutorials.mnist import input_data
#        mnist = input_data.read_data_sets('MNIST_data/', one_hot=False)
#        self.image = mnist.train.images
#        self.image = np.reshape(self.image, [len(self.image), 28, 28])
#        self.label = mnist.train.labels
#        self.noise = [np.random.normal(0., 1., 100) for i in xrange(len(self.image))]
#        self.noise = [noise / np.linalg.norm(noize) for noise in self.noise]
#        self.batch_idx = 0
#        self.rand_idx = np.random.permutation(len(self.image))
#
#    def __call__(self, color=True):
#        idx = self.rand_idx[self.batch_idx*self.batch_size : (self.batch_idx+1)*self.batch_size]
#
#        if (self.batch_idx+2)*self.batch_size > len(self.image)+1:
#            last_batch = True
#            self.batch_idx = 0
#            self.rand_idx = np.random.permutation(len(self.image))
#        else:
#            last_batch = False
#            self.batch_idx += 1
#
#        x, t, noise = self.image[idx], self.label[idx], self.noize[idx]
#        x = (x - 0.5) / 1.0
#        if color == True:
#            x = np.expand_dims(x, axis=3)
#            x = np.tile(x, (1, 1, 3))
#        return x, t, last_batch

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
