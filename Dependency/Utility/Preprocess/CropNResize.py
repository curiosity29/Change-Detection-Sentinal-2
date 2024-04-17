import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from tqdm import tqdm
class Crop_and_resize(tf.keras.layers.Layer):
  def __init__(self, seed = 42, crop_size = (256, 256), out_size = (256,256)):
    super().__init__()
    self.seed = seed
    
    if crop_size is None:
      self.cropper = None
    else:
      self.cropper = layers.RandomCrop(crop_size[1], crop_size[0], seed = seed)

    if crop_size == out_size:
      self.resizer = None
    else:
      self.resizer = layers.Resizing(out_size[1], out_size[0])

  def call(self, inputs, labels):
    label_type = labels.dtype
    # inputs, labels = tf.cast(inputs, tf.float32), tf.cast(labels, tf.float32)
    labels = tf.cast(labels, inputs.dtype)
    num_channels = inputs.shape[-1]
    stacked = tf.concat([inputs, labels], axis = -1)

    if self.cropper is not None:
      stacked = self.cropper(stacked)
    if self.resizer is not None:
      stacked = self.resizer(stacked)
    # output = self.resizer(rotated)

    # inputs = rotated[..., :num_channels]
    # labels = rotated[..., num_channels:]

    return stacked[..., :num_channels], tf.cast(stacked[..., num_channels:], dtype = label_type)


def usefulCut(image, label, cutter = Crop_and_resize(), drop_threshold = 0.8, drop_unlabeled_threshold = 0, drop_unlabeled_dim = False, check_ignore = False, **ignore):
  if check_ignore:
    print("ignored: ")
    print(ignore.keys())
  image, label = cutter(image, label)
  class_sums = np.sum(label, axis = (0,1))
  class_sum = np.sum(class_sums)
  # contain no unlabeled:
  if class_sums[0] > drop_unlabeled_threshold * class_sum:
    return None
  # useful threshold
  if any( class_sums > class_sum * drop_threshold):
    return None

  if drop_unlabeled_dim:
  # return label with removed unlabeled class
    return image, label[..., 1:]
  
  return image, label


def cutUntil(images, labels, usefulCut = usefulCut, size = (256, 256), n = 10, drop_threshold = 1.0, drop_unlabeled_threshold = 0, drop_unlabeled_dim = False, check_ignore = True, **ignore):
  """
    drop if too high in 1 class
    could run indefinately

    images and label shoud have the same size
  """
  if check_ignore:
    print("ignored: ")
    print(ignore.keys())
  length = len(images)
  result_images, result_labels = [], []
  count = 0
  cutter = Crop_and_resize(crop_size = size, out_size = size)
  length = len(images)
  index = 0 # index of the image in the list
  subindex = 0 # index of the cutted result in the current image
  with tqdm(total=n) as pbar:
    while count < n:
      image_, label_ = np.array(images[index]), np.array(labels[index])
      size_ratio = (image_.shape[0] * image_.shape[1]) // size[0] // size[1]
      # index = np.random.randint(0, length)
      cut = usefulCut(image_, label_, cutter, drop_threshold = drop_threshold, drop_unlabeled_threshold = drop_unlabeled_threshold, drop_unlabeled_dim = drop_unlabeled_dim, check_ignore = check_ignore, **ignore)
      if cut is None:
        continue

      count+=1
      image_, label_ = cut
      result_images.append(image_)
      result_labels.append(label_)
      subindex += 1

      pbar.update(1)
      # cut until an amount of times depend on the size, then next image
      if subindex >= size_ratio:
        subindex = 0
        index = (index + 1) % length


  return result_images, result_labels


def usefulCut_binary(image, label, cutter = Crop_and_resize(), drop_threshold = 5, **ignore):
  """
  drop if low pixel count for 1
  """
  image, label = cutter(image, label)
  class_sum = np.sum(label, axis = (0,1))
  # contain no unlabeled:
  # if drop_unlabeled and class_sum[0] > 0:
  #   return None
  # print(class_sum[0])
  # useful threshold
  if any( class_sum < drop_threshold):
    return None
  
  return image, label