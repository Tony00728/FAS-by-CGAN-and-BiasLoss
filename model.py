import tensorflow as tf
import ops
import utils
from reader import Reader
from discriminator import Discriminator
from generator import Generator
from VGGloss import vggloss
from sobel import sobel_gradient
from rgb_lab_formulation import rgb_to_lab

REAL_LABEL = 0.9
FAKE_LABEL = 0.1

class CycleGAN:
  def __init__(self,
               X_train_file='',
               Y_train_file='',
               batch_size=1,
               image_size=256,
               use_lsgan=True,
               norm='instance',
               lambda1=10.0,
               lambda2=10.0,
               learning_rate=2e-4,
               beta1=0.5,
               ngf=64
              ):
    """
    Args:
      X_train_file: string, X tfrecords file for training
      Y_train_file: string Y tfrecords file for training
      batch_size: integer, batch size
      image_size: integer, image size
      lambda1: integer, weight for forward cycle loss (X->Y->X)
      lambda2: integer, weight for backward cycle loss (Y->X->Y)
      use_lsgan: boolean
      norm: 'instance' or 'batch'
      learning_rate: float, initial learning rate for Adam
      beta1: float, momentum term of Adam
      ngf: number of gen filters in first conv layer
    """
    self.lambda1 = lambda1
    self.lambda2 = lambda2
    self.use_lsgan = use_lsgan
    use_sigmoid = not use_lsgan
    self.batch_size = batch_size
    self.image_size = image_size
    self.learning_rate = learning_rate
    self.beta1 = beta1
    self.X_train_file = X_train_file
    self.Y_train_file = Y_train_file
    self.is_training = tf.placeholder_with_default(True, shape=[], name='is_training')

    self.G = Generator('G', self.is_training, ngf=ngf, norm=norm, image_size=image_size)
    self.D_Y = Discriminator('D_Y',
        self.is_training, norm=norm, use_sigmoid=use_sigmoid)
    self.F = Generator('F', self.is_training, norm=norm, image_size=image_size)
    self.D_X = Discriminator('D_X',
        self.is_training, norm=norm, use_sigmoid=use_sigmoid)

    self.fake_x = tf.placeholder(tf.float32,
        shape=[batch_size, image_size, image_size, 3])
    self.fake_y = tf.placeholder(tf.float32,
        shape=[batch_size, image_size, image_size, 3])

  def model(self):
    X_reader = Reader(self.X_train_file, name='X',
        image_size=self.image_size, batch_size=self.batch_size)
    Y_reader = Reader(self.Y_train_file, name='Y',
        image_size=self.image_size, batch_size=self.batch_size)
    x = X_reader.feed()
    y = Y_reader.feed()

    cycle_loss = self.cycle_consistency_loss(self.G, self.F, x, y, self.D_Y)

    # X -> Y
    fake_y = self.G(x)
    G_gan_loss = self.generator_loss(self.D_Y, fake_y, y, use_lsgan=self.use_lsgan)
    G_loss =  G_gan_loss + cycle_loss

    D_Y_loss = self.discriminator_loss(self.D_Y, y, self.fake_y, use_lsgan=self.use_lsgan)

    # Y -> X
    fake_x = self.F(y)
    F_gan_loss = self.generator_loss(self.D_X, fake_x, y,  use_lsgan=self.use_lsgan)
    F_loss = F_gan_loss + cycle_loss
    D_X_loss = self.discriminator_loss(self.D_X, x, self.fake_x, use_lsgan=self.use_lsgan)

    # summary
    tf.summary.histogram('D_Y/true', self.D_Y(y))
    tf.summary.histogram('D_Y/fake', self.D_Y(self.G(x)))
    tf.summary.histogram('D_X/true', self.D_X(x))
    tf.summary.histogram('D_X/fake', self.D_X(self.F(y)))

#    tf.summary.scalar('loss/G', G_gan_loss)
    tf.summary.scalar('loss/D_Y', D_Y_loss)
#    tf.summary.scalar('loss/F', F_gan_loss)
    tf.summary.scalar('loss/D_X', D_X_loss)
    tf.summary.scalar('loss/cycle', cycle_loss)

    tf.summary.image('X/generated', utils.batch_convert2int(self.G(x)))
    tf.summary.image('X/reconstruction', utils.batch_convert2int(self.F(self.G(x))))
    tf.summary.image('Y/generated', utils.batch_convert2int(self.F(y)))
    tf.summary.image('Y/reconstruction', utils.batch_convert2int(self.G(self.F(y))))

    return G_loss, D_Y_loss, F_loss, D_X_loss, fake_y, fake_x

  def optimize(self, G_loss, D_Y_loss, F_loss, D_X_loss):
    def make_optimizer(loss, variables, name='Adam'):
      """ Adam optimizer with learning rate 0.0002 for the first 100k steps (~100 epochs)
          and a linearly decaying rate that goes to zero over the next 100k steps
      """
      global_step = tf.Variable(0, trainable=False)
      starter_learning_rate = self.learning_rate
      end_learning_rate = 0.0
      start_decay_step = 100000
      decay_steps = 100000
      beta1 = self.beta1
      learning_rate = (
          tf.where(
                  tf.greater_equal(global_step, start_decay_step),
                  tf.train.polynomial_decay(starter_learning_rate, global_step-start_decay_step,
                                            decay_steps, end_learning_rate,
                                            power=1.0),
                  starter_learning_rate
          )

      )
      tf.summary.scalar('learning_rate/{}'.format(name), learning_rate)

      learning_step = (
          tf.train.AdamOptimizer(learning_rate, beta1=beta1, name=name)
                  .minimize(loss, global_step=global_step, var_list=variables)
      )
      return learning_step

    G_optimizer = make_optimizer(G_loss, self.G.variables, name='Adam_G')
    D_Y_optimizer = make_optimizer(D_Y_loss, self.D_Y.variables, name='Adam_D_Y')
    F_optimizer =  make_optimizer(F_loss, self.F.variables, name='Adam_F')
    D_X_optimizer = make_optimizer(D_X_loss, self.D_X.variables, name='Adam_D_X')

    with tf.control_dependencies([G_optimizer, D_Y_optimizer, F_optimizer, D_X_optimizer]):
      return tf.no_op(name='optimizers')

  def discriminator_loss(self, D, y, fake_y, use_lsgan=True):
    """ Note: default: D(y).shape == (batch_size,5,5,1),
                       fake_buffer_size=50, batch_size=1
    Args:
      G: generator object
      D: discriminator object
      y: 4D tensor (batch_size, image_size, image_size, 3)
    Returns:
      loss: scalar
    """
    if use_lsgan:
      # use mean squared error
      #RGB
      #error_real_rgb = tf.reduce_mean(tf.squared_difference(D(y), REAL_LABEL))
      #error_fake_rgb = tf.reduce_mean(tf.square(D(fake_y)))
      #d1 = (error_real_rgb + error_fake_rgb)/2
      #HSV
      #y_hsv = tf.image.rgb_to_hsv(y)
      #fake_y_hsv = tf.image.rgb_to_hsv(fake_y)
      #error_real_hsv = tf.reduce_mean(tf.squared_difference(D(y_hsv), REAL_LABEL))
      #error_fake_hsv = tf.reduce_mean(tf.square(D(fake_y_hsv)))
      #d2 = (error_real_hsv + error_fake_hsv)/2
      #Lab
      error_real = tf.reduce_mean(tf.squared_difference(D(y), REAL_LABEL))
      error_fake = tf.reduce_mean(tf.square(D(fake_y)))    #original
      #error_fake = tf.reduce_mean(tf.squared_difference(D(fake_y), FAKE_LABEL))
    else:
      # use cross entropy
      error_real = -tf.reduce_mean(ops.safe_log(D(y)))
      error_fake = -tf.reduce_mean(ops.safe_log(1-D(fake_y)))
    loss = (error_real + error_fake) / 2
    return loss
############################################################################################### adjust G
  def generator_loss(self, D, fy, y, use_lsgan=True):
    """  fool discriminator into believing that G(x) is real
    """
    if use_lsgan:
      # use mean squared error
      #fy = scaleblur(fy)
      loss = tf.reduce_mean(tf.squared_difference(D(fy), REAL_LABEL))
      #Dloss = tf.reduce_mean(tf.squared_difference(D(y), FAKE_LABEL))
      #fy = tf.image.rgb_to_hsv(fy)
      #y = tf.image.rgb_to_hsv(y)
      #fy = rgb_to_lab(fy)
      #y = rgb_to_lab(y)

      vgg = vggloss(fy, y)    #vggloss  #要改
      #sobel = tf.reduce_mean(tf.squared_difference(sobel_gradient(fake_y), sobel_gradient(y)))
      #loss = (9*loss + Dloss)/10         #Dloss+genloss
      loss = loss + vgg/1000

    else:
      # heuristic, non-saturating loss
      loss = -tf.reduce_mean(ops.safe_log(D(fy))) / 2
    return loss
###############################################################################################

############################################################################################### adjust bias loss
  def cycle_consistency_loss(self, G, F, x, y, D_Y):
    """ cycle consistency loss (L1 norm)
    """
    #GF_hsv = tf.image.rgb_to_hsv(G(F(y)))
    #y_hsv = tf.image.rgb_to_hsv(y)
    #R1 = GF_hsv[:,:,:,0]
    #R2 = y_hsv[:,:,:,0]
    #G1 = GF_hsv[:,:,:,1]
    #G2 = y_hsv[:,:,:,1]
    #B1 = GF_hsv[:,:,:,2]
    #B2 = y_hsv[:,:,:,2]

    #R11 = F(G(x))[:,:,:,0]
    #R22 = y[:,:,:,0]

    #G11 = F(G(x))[:,:,:,1]
    #G22 = y[:,:,:,1]

    #B11 = F(G(x))[:,:,:,2]
    #B22 = y[:,:,:,2]


    #exloss1 =  tf.reduce_mean(tf.abs(B1-B2))

    #exloss2 = tf.reduce_mean(tf.abs(R11-R22)) + tf.reduce_mean(tf.abs(G11-G22)) + tf.reduce_mean(tf.abs(B11-B22))
    #exloss = exloss1 + exloss2
    #sb1 = G(x) + sobel_gradient(x)
    #sb2 = F(y) + sobel_gradient(y)
    #exloss = tf.reduce_mean(tf.abs(sb1 - x)) + tf.reduce_mean(tf.abs(sb2 - y))

    forward_loss = tf.reduce_mean(tf.abs(F(G(x))-x))
    backward_loss = tf.reduce_mean(tf.abs(G(F(y))-y))

    #bias_loss = tf.reduce_mean(tf.squared_difference(D_X(F(G(x))), FAKE_LABEL)) + tf.reduce_mean(tf.squared_difference(D_Y(G(F(y))), FAKE_LABEL))
    bias_loss = tf.reduce_mean(tf.squared_difference(D_Y(G(F(y))), FAKE_LABEL))

    #forward_loss = tf.reduce_mean(tf.squared_difference(F(G(x)), x))
    #backward_loss = tf.reduce_mean(tf.squared_difference(G(F(y)), y))
    #sobel = tf.reduce_mean(tf.abs(sobel_gradient(x)-sobel_gradient(y)))
    loss = self.lambda1 * forward_loss + self.lambda2 * backward_loss + bias_loss/10
    return loss

###############################################################################################


