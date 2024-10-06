import tensorflow as tf

## Layers: follow the naming convention used in the original paper
### Generator layers
def c7s1_k(input, k, reuse=False, norm='instance', activation='relu', is_training=True, name='c7s1_k'):
  """ A 7x7 Convolution-BatchNorm-ReLU layer with k filters and stride 1
  Args:
    input: 4D tensor
    k: integer, number of filters (output depth)
    norm: 'instance' or 'batch' or None
    activation: 'relu' or 'tanh'
    name: string, e.g. 'c7sk-32'
    is_training: boolean or BoolTensor
    name: string
    reuse: boolean
  Returns:
    4D tensor
  """
  with tf.variable_scope(name, reuse=reuse):
    weights = _weights("weights",
      shape=[7, 7, input.get_shape()[3], k])

    padded = tf.pad(input, [[0,0],[3,3],[3,3],[0,0]], 'REFLECT')
    conv = tf.nn.conv2d(padded, weights,
        strides=[1, 1, 1, 1], padding='VALID')

    normalized = _norm(conv, is_training, norm)

    if activation == 'relu':
      output = tf.nn.relu(normalized)
    if activation == 'tanh':
      output = tf.nn.tanh(normalized)
    return output

def dk(input, k, reuse=False, norm='instance', is_training=True, name=None):
  """ A 3x3 Convolution-BatchNorm-ReLU layer with k filters and stride 2
  Args:
    input: 4D tensor
    k: integer, number of filters (output depth)
    norm: 'instance' or 'batch' or None
    is_training: boolean or BoolTensor
    name: string
    reuse: boolean
    name: string, e.g. 'd64'
  Returns:
    4D tensor
  """
  with tf.variable_scope(name, reuse=reuse):
    weights = _weights("weights",
      shape=[3, 3, input.get_shape()[3], k])

    conv = tf.nn.conv2d(input, weights,
        strides=[1, 2, 2, 1], padding='SAME')
    normalized = _norm(conv, is_training, norm)
    output = tf.nn.relu(normalized)
    return output

def Rk(input, k,  reuse=False, norm='instance', is_training=True, name=None):
  """ A residual block that contains two 3x3 convolutional layers
      with the same number of filters on both layer
  Args:
    input: 4D Tensor
    k: integer, number of filters (output depth)
    reuse: boolean
    name: string
  Returns:
    4D tensor (same shape as input)
  """
  with tf.variable_scope(name, reuse=reuse):
    with tf.variable_scope('layer1', reuse=reuse):
      weights1 = _weights("weights1",
        shape=[3, 3, input.get_shape()[3], k])
      padded1 = tf.pad(input, [[0,0],[1,1],[1,1],[0,0]], 'REFLECT')
      conv1 = tf.nn.conv2d(padded1, weights1,
          strides=[1, 1, 1, 1], padding='VALID')
      normalized1 = _norm(conv1, is_training, norm)
      relu1 = tf.nn.relu(normalized1)

    with tf.variable_scope('layer2', reuse=reuse):
      weights2 = _weights("weights2",
        shape=[3, 3, relu1.get_shape()[3], k])

      padded2 = tf.pad(relu1, [[0,0],[1,1],[1,1],[0,0]], 'REFLECT')
      conv2 = tf.nn.conv2d(padded2, weights2,
          strides=[1, 1, 1, 1], padding='VALID')
      normalized2 = _norm(conv2, is_training, norm)
      relu2 = tf.nn.relu(normalized2)

    with tf.variable_scope('layer3', reuse=reuse):             #new resnet layer
      weights3 = _weights("weights3",
                          shape=[3, 3, relu2.get_shape()[3], k])

      padded3 = tf.pad(relu2, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
      conv3 = tf.nn.conv2d(padded3, weights3,
                           strides=[1, 1, 1, 1], padding='VALID')
      normalized3 = _norm(conv3, is_training, norm)
      relu3 = tf.nn.relu(normalized3)

    with tf.variable_scope('layer4', reuse=reuse):             #new resnet layer
      weights4 = _weights("weights4",
                          shape=[3, 3, relu3.get_shape()[3], k])

      padded4 = tf.pad(relu3, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
      conv4 = tf.nn.conv2d(padded4, weights4,
                           strides=[1, 1, 1, 1], padding='VALID')
      normalized4 = _norm(conv4, is_training, norm)
      relu4 = tf.nn.relu(normalized4)

    with tf.variable_scope('layer5', reuse=reuse):             #new resnet layer
      weights5 = _weights("weights5",
                          shape=[3, 3, relu4.get_shape()[3], k])

      padded5 = tf.pad(relu4, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
      conv5 = tf.nn.conv2d(padded5, weights5,
                           strides=[1, 1, 1, 1], padding='VALID')
      normalized5 = _norm(conv5, is_training, norm)
      relu5 = tf.nn.relu(normalized5)

    with tf.variable_scope('layer6', reuse=reuse):             #new resnet layer
      weights6 = _weights("weights6",
                          shape=[3, 3, relu5.get_shape()[3], k])

      padded6 = tf.pad(relu5, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
      conv6 = tf.nn.conv2d(padded6, weights6,
                           strides=[1, 1, 1, 1], padding='VALID')
      normalized6 = _norm(conv6, is_training, norm)
      relu6 = tf.nn.relu(normalized6)

    with tf.variable_scope('layer7', reuse=reuse):             #new resnet layer
      weights7 = _weights("weights7",
                          shape=[3, 3, relu6.get_shape()[3], k])

      padded7 = tf.pad(relu6, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
      conv7 = tf.nn.conv2d(padded7, weights7,
                           strides=[1, 1, 1, 1], padding='VALID')
      normalized7 = _norm(conv7, is_training, norm)
      relu7 = tf.nn.relu(normalized7)

    #with tf.variable_scope('layer8', reuse=reuse):             #new resnet layer
      #weights8 = _weights("weights8",
                          #shape=[3, 3, relu7.get_shape()[3], k])

      #padded8 = tf.pad(relu7, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
      #conv8 = tf.nn.conv2d(padded8, weights8,
                           #strides=[1, 1, 1, 1], padding='VALID')
      #normalized8 = _norm(conv8, is_training, norm)
      #relu8 = tf.nn.relu(normalized8)

    output = input + normalized2 + normalized3 + normalized4 + normalized5 + normalized6 + normalized7
    return output

def n_res_blocks(input, reuse, norm='instance', is_training=True, n=6):
  depth = input.get_shape()[3]
  for i in range(1,n+1):
    output = Rk(input, depth, reuse, norm, is_training, 'R{}_{}'.format(depth, i))
    input = output
  return output

def uk(input, k, reuse=False, norm='instance', is_training=True, name=None, output_size=None):
  """ A 3x3 Convolution-BatchNorm-ReLU layer with k filters and stride 2
  Args:
    input: 4D tensor
    k: integer, number of filters (output depth)
    norm: 'instance' or 'batch' or None
    is_training: boolean or BoolTensor
    reuse: boolean
    name: string, e.g. 'c7sk-32'
    output_size: integer, desired output size of layer
  Returns:
    4D tensor
  """
  with tf.variable_scope(name, reuse=reuse):
    input_shape = input.get_shape().as_list()

    weights = _weights("weights",
      shape=[3, 3, k, input_shape[3]])

    if not output_size:
      output_size = input_shape[1]*2
    output_shape = [input_shape[0], output_size, output_size, k]
    fsconv = tf.nn.conv2d_transpose(input, weights,
        output_shape=output_shape,
        strides=[1, 2, 2, 1], padding='SAME')
    normalized = _norm(fsconv, is_training, norm)
    output = tf.nn.relu(normalized)
    return output

### Discriminator layers
def Ck(input, k, slope=0.2, stride=2, reuse=False, norm='instance', is_training=True, name=None):
  """ A 4x4 Convolution-BatchNorm-LeakyReLU layer with k filters and stride 2
  Args:
    input: 4D tensor
    k: integer, number of filters (output depth)
    slope: LeakyReLU's slope
    stride: integer
    norm: 'instance' or 'batch' or None
    is_training: boolean or BoolTensor
    reuse: boolean
    name: string, e.g. 'C64'
  Returns:
    4D tensor
  """
  with tf.variable_scope(name, reuse=reuse):
    weights = _weights("weights",
      shape=[4, 4, input.get_shape()[3], k])

    conv = tf.nn.conv2d(input, weights,
        strides=[1, stride, stride, 1], padding='SAME')

    normalized = _norm(conv, is_training, norm)
    output = _leaky_relu(normalized, slope)
    return output

def last_conv(input, reuse=False, use_sigmoid=False, name=None):
  """ Last convolutional layer of discriminator network
      (1 filter with size 4x4, stride 1)
  Args:
    input: 4D tensor
    reuse: boolean
    use_sigmoid: boolean (False if use lsgan)
    name: string, e.g. 'C64'
  """
  with tf.variable_scope(name, reuse=reuse):
    weights = _weights("weights",
      shape=[4, 4, input.get_shape()[3], 1])
    biases = _biases("biases", [1])

    conv = tf.nn.conv2d(input, weights,
        strides=[1, 1, 1, 1], padding='SAME')
    output = conv + biases
    if use_sigmoid:
      output = tf.sigmoid(output)
    return output

### Helpers
def _weights(name, shape, mean=0.0, stddev=0.02):
  """ Helper to create an initialized Variable
  Args:
    name: name of the variable
    shape: list of ints
    mean: mean of a Gaussian
    stddev: standard deviation of a Gaussian
  Returns:
    A trainable variable
  """
  var = tf.get_variable(
    name, shape,
    initializer=tf.random_normal_initializer(
      mean=mean, stddev=stddev, dtype=tf.float32))
  return var

def _biases(name, shape, constant=0.0):
  """ Helper to create an initialized Bias with constant
  """
  return tf.get_variable(name, shape,
            initializer=tf.constant_initializer(constant))

def _leaky_relu(input, slope):
  return tf.maximum(slope*input, input)

def _norm(input, is_training, norm='instance'):
  """ Use Instance Normalization or Batch Normalization or None
  """
  if norm == 'instance':
    return _instance_norm(input)
  elif norm == 'batch':
    return _batch_norm(input, is_training)
  else:
    return input

def _batch_norm(input, is_training):
  """ Batch Normalization
  """
  with tf.variable_scope("batch_norm"):
    return tf.contrib.layers.batch_norm(input,
                                        decay=0.9,
                                        scale=True,
                                        updates_collections=None,
                                        is_training=is_training)

def _instance_norm(input):
  """ Instance Normalization
  """
  with tf.variable_scope("instance_norm"):
    depth = input.get_shape()[3]
    scale = _weights("scale", [depth], mean=1.0)
    offset = _biases("offset", [depth])
    mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
    epsilon = 1e-5
    inv = tf.rsqrt(variance + epsilon)
    normalized = (input-mean)*inv
    return scale*normalized + offset

def safe_log(x, eps=1e-12):
  return tf.log(x + eps)
