import numpy as np
from reader import read_cifar10

class cifar10(object):

  def __init__(self, bs=64):
    
    x, _ = read_cifar10('./cifar10_dataset', True).load_data()
    x2, _ = read_cifar10('./cifar10_dataset', False).load_data()
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
