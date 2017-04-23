import tensorflow as tf
import numpy as np
from cifar10 import cifar10
from model import GAN
from trainer import (Trainer, save_generated_examples)
import os
import sys
from config import *
import argparse

parser = argparse.ArgumentParser(description='Specify Training flag')
parser.add_argument('--training', type=bool, metavar='T')
args = parser.parse_args()

def main(training):
  
  with tf.Session() as sess:

    if training == True:
      if os.path.exists(summary_path):
        for fname in os.listdir(summary_path):
          os.remove(summary_path+'/'+fname)
      else:
        os.mkdir(path)

      cifar10_dataset = cifar10(bs=batch_size)
      gan = GAN()
      trainer = Trainer(gan, sess, cifar10_dataset, 200)
      init = tf.global_variables_initializer()

      sess.run(init)
      trainer.train()

    elif training == False:

      gan = GAN()
      saver = tf.train.Saver()
      saver = saver.restore(sess, model_save_path)
      image = sess.run(gan.x_hat, feed_dict={gan.z: np.random.uniform(-1, 1, (batch_size, num_z))})
      image = (image+1) / 2 

      save_generated_examples(image, 'examples') 

if __name__ == '__main__':
  main(args.training)