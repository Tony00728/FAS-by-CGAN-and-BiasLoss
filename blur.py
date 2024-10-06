import tensorflow as tf


with tf.variable_scope(name, reuse=reuse):
    with tf.variable_scope('layer1', reuse=reuse):
        weights1 = _weights("weights1",
                            shape=[3, 3, input.get_shape()[3], k])
        padded1 = tf.pad(input, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        conv1 = tf.nn.conv2d(padded1, weights1,
                             strides=[1, 1, 1, 1], padding='VALID')
        normalized1 = _norm(conv1, is_training, norm)
        relu1 = tf.nn.relu(normalized1)

    with tf.variable_scope('layer2', reuse=reuse):
        weights2 = _weights("weights2",
                            shape=[3, 3, relu1.get_shape()[3], k])

        padded2 = tf.pad(relu1, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        conv2 = tf.nn.conv2d(padded2, weights2,
                             strides=[1, 1, 1, 1], padding='VALID')
        normalized2 = _norm(conv2, is_training, norm)
    output = input + normalized2
    return output