"""Translate an image to another image
An example of command-line usage is:
python export_graph.py --model pretrained/apple2orange.pb \
                       --input input_sample.jpg \
                       --output output_sample.jpg \
                       --image_size 256
"""

import tensorflow as tf
import os
from model import CycleGAN
import utils

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('model', 'pretrained/young2old(50_MORPH_100000_2).pb', 'model path (.pb)')
tf.flags.DEFINE_string('input_folder', '/home/tony/age_code_testdataset/MORPH/31-40', 'input image path (.jpg)')
tf.flags.DEFINE_string('output_folder', '/home/tony/age_code_testdataset/MORPH/ours/31-40', 'output image path (.jpg)')
tf.flags.DEFINE_integer('image_size', '256', 'image size, default: 256')


def inference(input_image_path, output_image_path):
  graph = tf.Graph()

  with graph.as_default():
    with tf.gfile.FastGFile(input_image_path, 'rb') as f:
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
    with open(output_image_path, 'wb') as f:
      f.write(generated)


def main(unused_argv):
  input_folder = FLAGS.input_folder
  output_folder = FLAGS.output_folder

  # Ensure output folder exists
  if not os.path.exists(output_folder):
    os.makedirs(output_folder)

  # Process each image in the input folder
  for filename in os.listdir(input_folder):
    if filename.endswith(".JPG"):
      input_image_path = os.path.join(input_folder, filename)
      output_image_path = os.path.join(output_folder, filename)
      inference(input_image_path, output_image_path)


if __name__ == '__main__':
  tf.app.run()