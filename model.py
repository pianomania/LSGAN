import tensorflow as tf
from config import *

batch_norm = tf.contrib.layers.batch_norm
w_init = tf.contrib.layers.variance_scaling_initializer()
b_init = tf.constant_initializer(0.0)

def lrelu(x, leak=0.2, name="lrelu"):
     with tf.variable_scope(name):
         f1 = 0.5 * (1 + leak)
         f2 = 0.5 * (1 - leak)
         return f1 * x + f2 * abs(x)

class Generator(object):
  """ Generator """
  def __init__(self):
      
    y_depth = [256, 128, 128 , image_channel]
    batch_size = 64

    self.z = tf.placeholder(dtype=tf.float32, shape=[batch_size, num_z])

    y = tf.contrib.layers.fully_connected(
       inputs = self.z,
       num_outputs = 4*4*y_depth[0],
       normalizer_fn = batch_norm,
       normalizer_params = {'decay': decay_rate},
       weights_initializer = w_init,
       activation_fn = tf.nn.relu,
       scope = 'fc'
       )
    y = tf.reshape(y, (-1, 4, 4, y_depth[0]))
    
    y = tf.contrib.layers.convolution2d_transpose(
       inputs = y,
       num_outputs = y_depth[0],
       kernel_size = [3, 3],
       stride = [1, 1],
       padding = 'same',
       normalizer_fn = batch_norm,
       normalizer_params = {'decay': decay_rate},
       weights_initializer = w_init,
       activation_fn = tf.nn.relu,
       scope = 'conv_transposed0'
       )
    
    for i in range(0, 2):
      y = tf.contrib.layers.convolution2d_transpose(
         inputs = y,
         num_outputs = y_depth[i],
         kernel_size = [3, 3],
         stride = [2, 2],
         padding = 'same',
         normalizer_fn = batch_norm,
         normalizer_params = {'decay': decay_rate},
         weights_initializer = w_init,
         activation_fn = tf.nn.relu,
         scope = 'conv_transposed{}_{}'.format(i, 0)
         )    

      y = tf.contrib.layers.convolution2d_transpose(
         inputs = y,
         num_outputs = y_depth[i],
         kernel_size = [3, 3],
         stride = [1, 1],
         padding = 'same',
         normalizer_fn = batch_norm,
         normalizer_params = {'decay': decay_rate},
         weights_initializer = w_init,
         activation_fn = tf.nn.relu,
         scope = 'conv_transposed{}_{}'.format(i, 1)
         )    
    y3 = y
    
    y4 = tf.contrib.layers.convolution2d_transpose(
       inputs = y3,
       num_outputs = y_depth[-1],
       kernel_size = [3, 3],
       stride = [2, 2],
       padding = 'same',
       weights_initializer = w_init,
       activation_fn = tf.tanh,
       scope = 'conv_transposed_final'
       )
    
    self.x_hat = y4

class Discriminator(object):
  """ Discriminator """
  def __init__(self, reuse, x=None):
    
    if reuse == False:
      self.image = tf.placeholder(tf.float32, shape=[64, 32, 32, image_channel])
    else:
      self.image = x

    y0 = tf.contrib.layers.conv2d(
        self.image,
        num_outputs = 64,
        kernel_size = 5,
        stride = (2, 2),
        padding = 'same',
        weights_initializer = w_init,
        activation_fn = lrelu,
        reuse = reuse,
        scope = 'conv0'
      )

    y1 = tf.contrib.layers.conv2d(
        y0,
        num_outputs = 128,
        kernel_size = 5,
        stride = (2, 2),
        padding = 'same',
        normalizer_fn = batch_norm,
        normalizer_params = {'decay': decay_rate},
        weights_initializer = w_init,
        activation_fn = lrelu,
        reuse = reuse,
        scope = 'conv1'
      )

    y2 = tf.contrib.layers.conv2d(
        y1,
        num_outputs = 128,
        kernel_size = 5,
        stride = (2, 2),
        padding = 'same',
        normalizer_fn = batch_norm,
        normalizer_params = {'decay': decay_rate},
        weights_initializer = w_init,
        activation_fn = lrelu,
        reuse = reuse,
        scope = 'conv2'
      )
    
    y3 = tf.contrib.layers.conv2d(
        y2,
        num_outputs = 256,
        kernel_size = 5,
        stride = (2, 2),
        padding = 'same',
        normalizer_fn = batch_norm,
        normalizer_params = {'decay': decay_rate},
        weights_initializer = w_init,
        activation_fn = lrelu,
        reuse = reuse,
        scope = 'conv3'
      )
    
    to_output = y3

    logits = tf.contrib.layers.conv2d(
        to_output,
        num_outputs = 1,
        kernel_size = to_output.get_shape().as_list()[1],
        stride = (1, 1),
        padding = 'valid',
        weights_initializer = w_init,
        activation_fn = None,
        reuse = reuse,
        scope = 'output_layer'
      )

    self.y = logits

class Gmlp(object):

  def __init__(self, nlayers):
    
    self.z = tf.placeholder(dtype=tf.float32, shape=[64, 100])

    for i in range(nlayers+1):
      y = tf.contrib.layers.fully_connected(
            inputs = self.z if i == 0 else y,
            num_outputs = 512 if i != nlayers else 784,
            activation_fn = tf.nn.relu if i != nlayers else tf.sigmoid,
            weights_initializer = w_init,
            scope = 'Gmlp{}'.format(i)
          )

    self.x_hat = y

class Dmlp(object):

  def __init__(self, reuse, nlayers, x=None):

    if reuse == False:
      self.image = tf.placeholder(tf.float32, shape=[64, 784])
    else:
      self.image = x

    for i in range(nlayers+1):
      y = tf.contrib.layers.fully_connected(
            inputs = self.image if i == 0 else y,
            num_outputs = 512 if i != nlayers else 1,
            activation_fn = lrelu if i != nlayers else None,
            weights_initializer = w_init,
            scope = 'Dmlp{}'.format(i) 
          )

    self.y = tf.squeeze(y)

class GAN(object):

  def __init__(self):

    with tf.variable_scope('GAN'):

      with tf.variable_scope('Generator'):
        self.G = Generator()
        self.z = self.G.z
        self.x_hat = self.G.x_hat

      with tf.variable_scope('Discriminator'):
        self.D_real = Discriminator(False) 
        self.real_x = self.D_real.image

        self.D_fake = Discriminator(True, self.x_hat)

      self.d_loss = 0.5*(tf.reduce_mean(tf.square(self.D_real.y-1.)) + tf.reduce_mean(tf.square(self.D_fake.y)))
      self.g_loss = 0.5*tf.reduce_mean(tf.square(self.D_fake.y-1.))
