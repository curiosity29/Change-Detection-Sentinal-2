import tensorflow as tf
from tensorflow.keras import layers

from Blocks.Basic import UNet_encoder, UNet_decoder, CoBaRe

def UNet(n_channel = 3, n_class = 2, input_size = 256, output_size = 256, softmax_head = True, filters = (64, 128, 256, 512), **ignore):
  inputs = tf.keras.Input(shape = (input_size, input_size, n_channel), name = "UNet_input")

  x, skips = UNet_encoder(inputs, filters)

  x = CoBaRe(filters = 1024, name = "bottleneck_1")(x)
  x = CoBaRe(filters = 1024, name = "bottleneck_2")(x)

  x = UNet_decoder(x, list(reversed(filters)), skips)
  
  x = layers.Conv2D(filters = n_class, kernel_size = 1, padding = "same", name = "segment_head")(x)
  if softmax_head:
    x = layers.Activation("softmax", name = "softmax_head")(x)

  return tf.keras.Model(inputs, x)
