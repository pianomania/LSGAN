import tensorflow as tf
import numpy as np
from load_data import load_data
from model import GAN
from trainer import (Trainer, save_generated_examples)
import os
import sys
from config import *
import argparse

parser = argparse.ArgumentParser(description='Specify Training flag')
parser.add_argument('--training', type=int, metavar='T')
parser.add_argument('--dataset', type=str, metavar='S')
args = parser.parse_args()

def main(args):
  
  with tf.Session() as sess:

    training = args.training
    dataset_name = args.dataset


    if training == 1:
      if os.path.exists(summary_path):
        for fname in os.listdir(summary_path):
          os.remove(summary_path+'/'+fname)
      else:
        os.mkdir(path)

      dataset = load_data(dataset_name, bs=batch_size)
      gan = GAN()
      trainer = Trainer(gan, sess, dataset, 200)
      init = tf.global_variables_initializer()

      sess.run(init)
      trainer.train()

    elif training == 0:

      gan = GAN()
      saver = tf.train.Saver()
      saver = saver.restore(sess, model_save_path)
      image = sess.run(gan.x_hat, feed_dict={gan.z: np.random.uniform(-1, 1, (batch_size, num_z))})
      image = (image+1) / 2 

      save_generated_examples(image, 'examples') 

if __name__ == '__main__':
  main(args)