import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from config import *

class Trainer(object):

  def __init__(self, GAN, sess, dataset, n_epoch):

    self.sess = sess
    self.GAN = GAN
    self.dataset = dataset
    self.n_epoch = n_epoch

    var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    D_vars = [v for v in var if 'Discriminator' in v.name]
    G_vars = [v for v in var if 'Generator' in v.name]

    self.D_vars = D_vars
    self.G_vars = G_vars

    with tf.variable_scope('Trainer'):
      self.D_global_step = tf.Variable(0.0, trainable=False, name='D_global_step')
      self.G_global_step = tf.Variable(0.0, trainable=False, name='G_global_step')

      self.D_lr = tf.train.polynomial_decay(0.0002, self.G_global_step, 200000, 0.00002)
      self.G_lr = tf.train.polynomial_decay(0.001, self.G_global_step, 200000, 0.0001)
      self.optD = tf.train.AdamOptimizer(self.D_lr, name='D_Optimizer', epsilon=1e-6, beta1=0.5, beta2=0.99)
      self.optG = tf.train.AdamOptimizer(self.G_lr, name='G_Optimizer', epsilon=1e-6, beta1=0.5, beta2=0.99)      
      

      self.D_gvs = self.optD.compute_gradients(GAN.d_loss, D_vars)
      self.D_backprop = self.optD.apply_gradients(self.D_gvs, self.D_global_step)

      self.G_gvs = self.optG.compute_gradients(GAN.g_loss, G_vars)
      self.G_backprop = self.optG.apply_gradients(self.G_gvs, self.G_global_step)

  def _summary(self):

    tf.summary.scalar('D_loss', self.GAN.d_loss)
    tf.summary.scalar('G_loss', self.GAN.g_loss)
    tf.summary.histogram('real_y', self.GAN.D_real.y)
    tf.summary.histogram('fake_y', self.GAN.D_fake.y)
    tf.summary.image('generated image', self.GAN.x_hat, max_outputs=64)

    #for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
    #  if 'Trainer' not in v.name and 'weights' or 'bias' in v.name:
    #    tf.summary.histogram(v.name, v)
  
    self.summary_writer = tf.summary.FileWriter(summary_path, self.sess.graph)
    self.summary_op = tf.summary.merge_all()

  def train(self):

    self._summary()
    saver = tf.train.Saver()

    num_iters = 0

    fix_noise = np.random.uniform(-1, 1, (batch_size, num_z))

    for i in range(self.n_epoch):
      for iteration in range(self.dataset.iter_per_epoch):
        x = self.dataset.batch()
        z = np.random.uniform(-1, 1, (batch_size, num_z))

        dloss, _ = self.sess.run([self.GAN.d_loss, self.D_backprop], 
          feed_dict={self.GAN.real_x: x,
                     self.GAN.z: z})

        z = np.random.uniform(-1, 1, (batch_size, num_z))
        gloss, _ = self.sess.run([self.GAN.g_loss, self.G_backprop],
          feed_dict={self.GAN.z: z})
        
        num_iters += 1

        if num_iters % 100 == 0:

          x_hat, summary = self.sess.run(
            [self.GAN.x_hat, self.summary_op],
            feed_dict={self.GAN.real_x: x,
                       self.GAN.z: fix_noise})

          self.summary_writer.add_summary(summary, num_iters)

          print 'iter:{0}, Dloss: {1:.6f}, Gloss: {2:.6f}'.format(num_iters, dloss, gloss)

          if num_iters % 1000 == 0:
            saver.save(self.sess, model_save_path)

            x_hat = (x_hat + 1) / 2
            save_generated_examples(x_hat, str(num_iters))

def save_generated_examples(image, name):
  k=0
  fig, ax =plt.subplots(8,8)
  for i in xrange(8):
    for j in xrange(8):
      ax[i][j].imshow(image[k])
      ax[i][j].set_axis_off()
      k += 1
  plt.tight_layout(h_pad=-1.5,w_pad=-14)
  plt.savefig(assets_path+ name + '.png')
  plt.close()