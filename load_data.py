import numpy as np
from reader import read_cifar10
from scipy.io import loadmat

class load_data(object):

  def __init__(self, name=None, bs=64):
    
    if name == 'svhn':
        data = loadmat('./dataset/svhn/svhn.mat')
        self.images = data['X'].transpose(3,0,1,2)/127.5 - 1.0
        self.nums_images = self.images.shape[0]

    elif name == 'cifar10':
        x, _ = read_cifar10('./dataset/cifar10', True).load_data()
        x2, _ = read_cifar10('./dataset/cifar10', False).load_data()
        x = np.vstack([x, x2])
        self.images = np.transpose(x.reshape(-1,3,32,32), (0,2,3,1))/127.5 - 1.0
        self.nums_images = self.images.shape[0]
    
    self.batch_size = bs
    self.iter_per_epoch = self.nums_images/bs
    self._generate_idx()

  def _generate_idx(self):
    gen_idx = np.random.permutation(self.nums_images)[0:self.iter_per_epoch*64]
    self.batch_in_epoch = np.split(gen_idx, self.iter_per_epoch)
    self.trace_idx = 0

  def batch(self):

    if self.trace_idx == self.iter_per_epoch:
      self._generate_idx()

    idx = self.batch_in_epoch[self.trace_idx]
    x = self.images[idx]
    self.trace_idx += 1

    return x
