import tensorflow as tf

def sobel_gradient(im):
    '''
    Args:
    im: Image to be differentiated [B,H,W,3]
    Returns:
    grad: Sobel gradient magnitude of input image [B,H,W,1]
    '''

    assert im.get_shape()[-1].value == 3
    print(im)

    Gx_kernel = tf.tile(tf.constant([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], shape=[3, 3, 1, 1], dtype=tf.float32),
                        [1, 1, 3, 1])
    Gy_kernel = tf.transpose(Gx_kernel, [1, 0, 2, 3])

    Gx = tf.nn.conv2d(im, Gx_kernel, [1, 1, 1, 1], padding='SAME')
    Gy = tf.nn.conv2d(im, Gy_kernel, [1, 1, 1, 1], padding='SAME')

    grad = tf.sqrt(tf.add(tf.pow(Gx, 2), tf.pow(Gy, 2)))
    grad = tf.truediv(grad, 3.)
    grad = tf.image.grayscale_to_rgb(grad)
    return grad

def rgb_to_yuv(images):
  """Converts one or more images from RGB to YUV.
  Outputs a tensor of the same shape as the `images` tensor, containing the YUV
  value of the pixels.
  The output is only well defined if the value in images are in [0,1].
  Args:
    images: 2-D or higher rank. Image data to convert. Last dimension must be
    size 3.
  Returns:
    images: tensor with the same shape as `images`.
  """
  images = ops.convert_to_tensor(images, name='images')
  kernel = ops.convert_to_tensor(
      _rgb_to_yuv_kernel, dtype=images.dtype, name='kernel')
  ndims = images.get_shape().ndims
  return math_ops.tensordot(images, kernel, axes=[[ndims - 1], [0]])


_yuv_to_rgb_kernel = [[1, 1, 1], [0, -0.394642334, 2.03206185],
                      [1.13988303, -0.58062185, 0]]