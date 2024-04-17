import tensorflow as tf
from tensorflow.keras import layers
from keras import ops
import numpy as np

class Patches(tf.keras.layers.Layer):


  def __init__(self, patch_size):
    super().__init__()
    self.patch_size = patch_size

  def call(self, image):

    input_shape = ops.shape(image)
    batch_size, height, width, channels = input_shape
    num_patch_h, num_patch_w = height//self.patch_size, width//self.patch_size
    patches = ops.image.extract_patches(image, size = self.patch_size)

    patches = ops.reshape(
        patches,
        (
        batch_size,
        num_patch_h * num_patch_w,
        self.patch_size*self.patch_size * channels
        )
      )
    # patches = ops.reshape(
    #     patches,
    #     (batch_size, num_patch_w * self.patch_size, num_patch_w * self.patch_size, 4)
    # )
    return patches

  def get_config(self):

    config = super().get_config()
    config.update({
        'patch_size': self.patch_size,
    })
    return config

class UnPatches(tf.keras.layers.Layer):

  def __init__(self, patch_size):
    super().__init__()
    self.patch_size = patch_size

  def call(self, image):
    batch_size, num_patches, dim = ops.shape(image)
    out_size = int(np.sqrt(num_patches)) * self.patch_size
    # out_dim = image.shape[-1] // patch_size**2
    out_dim = dim // self.patch_size**2
    # unpatches = ops.reshape(
    #     image,
    #     (image.shape[0], num_patch_w * self.patch_size, num_patch_w * self.patch_size, out_dim)
    # )
    unpatches = ops.reshape(
        image,
        (batch_size, out_size, out_size, out_dim)
    )

    return unpatches

  def get_config(self):

    config = super().get_config()
    config.update({
        'patch_size': self.patch_size,
    })
    return config

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = ops.expand_dims(
            ops.arange(start=0, stop=self.num_patches, step=1), axis=0
        )
        projected_patches = self.projection(patch)
        encoded = projected_patches + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches})
        return config



