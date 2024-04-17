import tensorflow as tf
from functools import partial
import os
from tqdm import tqdm
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


def file_iterate_name(idx, root, name = "train"):
  return os.path.join(root, f"AT_record_{name}_{idx}")

def file_iterate_auto(name = "train"):
  return partial(file_iterate_name, name = name)

def create_name_dataset(n, root, file_iterate):
## data to read
  if not callable(file_iterate):
    file_iterate = file_iterate_auto(name = str(file_iterate))
  # reading with string datasret
  filenames = []

  for idx in range(n):
    filename = file_iterate(idx, root)
    filenames.append(filename)
  return tf.data.TFRecordDataset(filenames)


def create_records(images, masks, root, file_iterate = None, start_index = 0):
  if not callable(file_iterate):
    file_iterate = file_iterate_auto(name = str(file_iterate))
  n = len(images)
  for idx in tqdm(range(n)):
    with tf.io.TFRecordWriter(file_iterate(start_index + idx, root)) as writer:
      example = seri(images[idx], masks[idx])
      writer.write(example)
  print(f"created from {start_index} to {start_index + n}")
  return file_iterate


def read(record, image_dtype = tf.float32, mask_dtype = tf.uint16):
  features = {
      "image": tf.io.FixedLenFeature([], tf.string),
      "mask": tf.io.FixedLenFeature([], tf.string),
  }
  example = tf.io.parse_single_example(record, features)
  image = tf.io.parse_tensor(example["image"], out_type=image_dtype)
  mask = tf.io.parse_tensor(example["mask"], out_type=mask_dtype)
  return image, mask