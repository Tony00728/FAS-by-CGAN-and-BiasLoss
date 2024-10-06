import tensorflow as tf
import os
from model import CycleGAN
import utils

path = '/home/tony/ours/test_all/MORPH_51+'

d_path = '/home/tony/age_code/paper_fig/test_MORPH_51+'
if not os.path.isdir(d_path):
    os.makedirs(d_path)

files = os.listdir(path)
num_files = len(files)

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('model', 'pretrained/old2young(30vgg16).pb', 'model path (.pb)')
tf.flags.DEFINE_integer('image_size', '256', 'image size, default: 256')


for i in range(num_files):

    inim = path + '/' + files[i]
    outim = d_path + '/' + files[i]

    # tf.flags.DEFINE_string('input', inim, 'input image path (.jpg)')
    # tf.flags.DEFINE_string('output', outim, 'output image path (.jpg)')

    graph = tf.Graph()

    with graph.as_default():
        with tf.gfile.FastGFile(inim, 'rb') as f:
          image_data = f.read()
          input_image = tf.image.decode_jpeg(image_data, channels=3)
          input_image = tf.image.resize_images(input_image, size=(FLAGS.image_size, FLAGS.image_size))
          input_image = utils.convert2float(input_image)
          input_image.set_shape([FLAGS.image_size, FLAGS.image_size, 3])

        with tf.gfile.FastGFile(FLAGS.model, 'rb') as model_file:
          graph_def = tf.GraphDef()
          graph_def.ParseFromString(model_file.read())
        [output_image] = tf.import_graph_def(graph_def,
                              input_map={'input_image': input_image},
                              return_elements=['output_image:0'],
                              name='output')

    with tf.Session(graph=graph) as sess:
        generated = output_image.eval()
        with open(outim, 'wb') as f:
          f.write(generated)

        print(i)