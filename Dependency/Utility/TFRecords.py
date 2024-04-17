from tokenize import Name
import tensorflow as tf

from functools import partial
import os
import glob

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def seri(image, mask):
  image = tf.io.serialize_tensor(image)
  mask = tf.io.serialize_tensor(mask)
  feature = {
      "image": _bytes_feature(image),
      "mask": _bytes_feature(mask)
  }
  return tf.train.Example(features = tf.train.Features(feature = feature)).SerializeToString()


def file_iterate_name(root, idx, name = "train"):
  return os.path.join(root, f"AT_record_{name}_{idx}")

def file_iterate_possible(root, idx):
  return glob.glob(os.path.join(root, "*idx*")[0])

def file_iterate_auto(name = "train"):
  return partial(file_iterate_name, name = name)
# def file_iterate_test(root, idx):
#   return f"{root}AT_record_test_{idx}"

def create_name_dataset(n, root, file_iterate):
## data to read

  # reading with string dataset
  filenames = []

  for idx in range(n):
    filename = file_iterate(root, idx)
    filenames.append(filename)
  return tf.data.TFRecordDataset(filenames)

def create_records(images, masks, file_iterate, root, start_index = 0):
  if not callable(file_iterate):
    file_iterate = file_iterate_auto(name = file_iterate)
  n = len(images)
  for idx in range(n):
    with tf.io.TFRecordWriter(file_iterate(root, idx)) as writer:
      example = seri(images[idx], masks[idx])
      writer.write(example)

  return file_iterate

def read(record):
  features = {
      "image": tf.io.FixedLenFeature([], tf.string),
      "mask": tf.io.FixedLenFeature([], tf.string),
  }
  example = tf.io.parse_single_example(record, features)
  image = tf.io.parse_tensor(example["image"], out_type=tf.float32)
  mask = tf.io.parse_tensor(example["mask"], out_type=tf.uint16)
  return image, mask