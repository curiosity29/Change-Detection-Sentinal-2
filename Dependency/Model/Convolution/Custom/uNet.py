import tensorflow as tf
from tensorflow.keras import layers

# from Blocks.Basic import UNet_encoder, UNet_decoder, CoBaRe
from Blocks.Basic import *


def uNet1(n_channel = 3, n_class = 2, input_size = 256, output_size = 256, head = "sigmoid", filters = (16, 32, 64, 128), **ignore):
  inputs = tf.keras.Input(shape = (input_size, input_size, n_channel), name = "UNet_input")

  x, skips = UNet_encoder(inputs, filters)

  x = CoBaRe(filters = 256, name = "bottleneck_1")(x)
  x = CoBaRe(filters = 256, name = "bottleneck_2")(x)

  x = UNet_decoder(x, list(reversed(filters)), skips)
  
  x = layers.Conv2D(filters = n_class, kernel_size = 1, padding = "same", name = "segment_head")(x)
  if head is not None:
    x = layers.Activation(head, name = f"{head}_head")(x)

  return tf.keras.Model(inputs, x)

def dilatedUNet1(n_channel = 3, n_class = 2, input_size = 256, output_size = 256, filters = (16, 32, 64, 128), head = "sigmoid", **ignore):
  inputs = tf.keras.Input(shape = (input_size, input_size, n_channel), name = "UNet_input")

  x, skips = dilated_UNet_encoder(inputs, filters)

  x = CoBaRe(filters = 256, name = "bottleneck_1")(x)
  x = CoBaRe(filters = 256, name = "bottleneck_2")(x)

  x = UNet_decoder(x, list(reversed(filters)), skips)
  
  x = layers.Conv2D(filters = n_class, kernel_size = 1, padding = "same", name = "segment_head")(x)
  if head is not None:
    x = layers.Activation(head, name = f"{head}_head")(x)

  return tf.keras.Model(inputs, x)
