"""Converts VGG data to TFRecords file format."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import tensorflow as tf
import numpy as np
import skimage.io as io
from collections import namedtuple

FLAGS = None

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def one_hot(input, dict):
    num_example = input.shape[0]
    dim = len(dict)
    output = np.zeros((num_example, dim))
    tokenized_input = [dict.index(x) for x in input]
    output[np.arange(num_example), tokenized_input] = 1
    return output

#input numpy array, output list
def tokenize(input, dict):
    return [dict.index(x) for x in input]

def readImg(data_folder, val_split, test_split):
    rawData = namedtuple('rawData', ['train', 'validation', 'test'])
    dataSet = namedtuple('dataSet', ['images', 'labels', 'num_examples'])
    data = []
    labels = []
    cnt = 0
    label_dict = []
    for person in os.listdir(data_folder):
        label_dict.append(person)
        path = os.path.join(data_folder, person)
        for image in os.listdir(path):
            raw_img = io.imread(os.path.join(path, image))
            data.append(raw_img)
            labels.append(person)
            cnt += 1
    #shuffle input examples
    indices = np.arange(cnt)
    np.random.shuffle(indices)
    data = np.array(data)
    data = data[indices]
    labels = np.array(labels)
    labels = labels[indices]

    #tokenize labels
    labels = tokenize(labels, label_dict)

    #split data into training set, validation set and test set
    train_num = int(round((1 - val_split - test_split)* cnt))
    val_num = int(round(val_split * cnt))
    test_num = int(round(test_split * cnt))
    train_input = data[:train_num]
    train_labels = labels[:train_num]
    val_input = data[train_num:(train_num + val_num)]
    val_labels = labels[train_num:(train_num + val_num)]
    test_input = data[(train_num + val_num):]
    test_labels = labels[(train_num + val_num):]

    #build data set
    VGG_train = dataSet(train_input, train_labels, train_num)
    VGG_val = dataSet(val_input, val_labels, val_num)
    VGG_test = dataSet(test_input, test_labels, test_num)
    return rawData(VGG_train, VGG_val, VGG_test)

def convert_to(data_set, name):
  """Converts a dataset to tfrecords."""
  images = data_set.images
  labels = data_set.labels
  num_examples = data_set.num_examples

  if images.shape[0] != num_examples:
    raise ValueError('Images size %d does not match label size %d.' %
                     (images.shape[0], num_examples))
  rows = images.shape[1]
  cols = images.shape[2]
  depth = images.shape[3]

  filename = os.path.join(FLAGS.directory, name + '.tfrecords')
  print('Writing', filename)
  writer = tf.python_io.TFRecordWriter(filename)
  for index in range(num_examples):
    image_raw = images[index].tostring()
    #example = tf.train.Example(features=tf.train.Features(feature={
        #'height': _int64_feature(rows),
        #'width': _int64_feature(cols),
        #'depth': _int64_feature(depth),
        #'label': _int64_feature(int(labels[index])),
        #'image_raw': _bytes_feature(image_raw)}))
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': _bytes_feature(image_raw),
        'image/format': _bytes_feature('png'),
        'image/class/label': _int64_feature(int(labels[index])),
        'image/height': _int64_feature(rows),
        'image/width': _int64_feature(cols)}))
    writer.write(example.SerializeToString())
  writer.close()

def convert_back_and_check(data_set):
    reconstructed_images = []
    filename = os.path.join(FLAGS.directory, 'train.tfrecords')
    record_iterator = tf.python_io.tf_record_iterator(filename)
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)

        encoded_img = (example.features.feature['image/encoded']
                     .bytes_list
                     .value[0])

        format = (example.features.feature['image/format']
                    .bytes_list
                    .value[0])

        label = (example.features.feature['image/class/label']
                      .int64_list
                      .value[0])

        height = int(example.features.feature['image/height']
                             .int64_list
                             .value[0])

        width = int(example.features.feature['image/width']
                  .int64_list
                  .value[0])

        img_1d = np.fromstring(encoded_img, dtype=np.uint8)
        reconstructed_img = img_1d.reshape((height, width, -1))
        reconstructed_images.append(reconstructed_img)

    for pair in zip(data_set.images, reconstructed_images):
        print(np.allclose(*pair))


def main(unused_argv):
  # Get the data.
  aligned_folder = os.path.join(FLAGS.directory, 'aligned/')
  data_sets = readImg(aligned_folder, FLAGS.validation_split, FLAGS.test_split)

  # Convert to Examples and write the result to TFRecords.
  convert_to(data_sets.train, 'train')
  convert_to(data_sets.validation, 'validation')
  convert_to(data_sets.test, 'test')
  convert_back_and_check(data_sets.train)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--directory',
      type=str,
      default='/home/tjiang/PycharmProjects/face_recogition/vgg_face_dataset/',
      help='Directory to download data files and write the converted result'
  )
  parser.add_argument(
      '--validation_split',
      type=float,
      default=0.1,
      help="""\
      Percentage of validation data
      set.\
      """
  )
  parser.add_argument(
      '--test_split',
      type=float,
      default=0.2,
      help="""\
      Percentage of test data
      set.\
      """
  )
  FLAGS, unparsed = parser.parse_known_args()
tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
