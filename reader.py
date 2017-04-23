import numpy as np
import os
import re
import cPickle

class read_cifar10(object):

  def __init__(self, data_path=None, is_training=True):
    self.data_path = data_path
    self.is_training = is_training

  def load_data(self):

    files = os.listdir(self.data_path)

    if self.is_training is True:
      pattern = re.compile('(data_batch_).')

      to_read = [m.group(0) for i in files for m in [pattern.search(i)] if m] 

      data = []
      labels = []

      for t in to_read:
        with open(self.data_path+'/'+t, 'rb') as f:
          d = cPickle.load(f)
          data.append(d['data'])
          labels.append(d['labels'])

      data = np.vstack(data)
      labels = np.hstack(labels)

    else:
      with open(self.data_path+'/test_batch') as f:
        d = cPickle.load(f)
        data = d['data']
        labels = d['labels']
    
    return data, labels

