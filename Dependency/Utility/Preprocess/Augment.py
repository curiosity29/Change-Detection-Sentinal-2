import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import tensorflow.keras.backend as K
from General.Basic import poly_to_mask
from functools import partial
import cv2
# import tensorflow_models as tfm
""" To do:

add probability to random layer

""" 
class Augment1(tf.keras.layers.Layer):
  def __init__(self, seed = 42, size = (256,256), rotate = None, noise = None, 
  zoom = None, brightness = None, offset = None, shade = None, 
  contrast = None, saturation = None,
  probs = [0.5]*5, **ignore):
    super().__init__()
    self.probs = probs
    self.seed = seed
    self.size = size
    self.cropper = layers.RandomCrop(size[0], size[1], seed = seed)
    self.rotate = rotate
    if rotate is not None:
      self.rotator = layers.RandomRotation(rotate, seed = seed)
    self.noise = noise
    if noise is not None:
      self.noiser = layers.GaussianNoise(noise, seed = seed)

    self.zoom = zoom
    if zoom is not None:
      self.zoomer = layers.RandomZoom(*zoom, seed = seed)
    
    self.brightness = brightness
    if brightness is not None:
      self.brighter = layers.RandomBrightness(brightness, value_range = [0.0, 1.0], seed = seed)

    self.offset = offset
    if offset is not None:
      self.centerer = layers.CenterCrop(size[0] - offset, size[1] - offset)

    self.shade = shade
    if shade is not None:
      self.shade = int(shade)
      self.shader = RandomShade(n_shadow=self.shade)
    
    self.contrast = contrast
    if contrast is not None:
      self.contraster = layers.RandomContrast(contrast)

    self.saturation = saturation
    if saturation is not None:
      self.saturater = RandomSaturation(saturation)


  def call(self, inputs, labels):
    inputs, labels = tf.cast(inputs, tf.float32), tf.cast(labels, tf.float32)
    # keeps = np.random.uniform(0,1,size = 5) < self.probs
    if self.noise is not None:
      inputs = self.noiser(inputs, training = True)
    if self.brightness is not None:
      inputs = self.brighter(inputs, training = True)
    if self.contrast is not None:
      inputs = self.contraster(inputs, training = True)
    if self.saturation is not None:
      ## apply saturation to the first 3 channels
      ## (random saturation only work for rgb channels)
      
      # inputs[..., :3] = self.saturater(inputs[..., :3], training = True)
      saturated_3 = self.saturater(inputs[..., :3], training = True)
      
      inputs = tf.concat([saturated_3, inputs[..., 3:]], axis = -1)
    if self.shade is not None:
      inputs = self.shader(inputs, training = True)

    num_channels = inputs.shape[-1]
    stacked = tf.concat([inputs, labels], axis = -1)
    # cropped = self.cropper(stacked)
    if self.rotate is not None:
      stacked = self.rotator(stacked, training = True)

    if self.zoom is not None:
      stacked = self.zoomer(stacked, training = True)


    if self.offset is not None:
      labels = self.centerer(tf.cast(stacked[..., num_channels:], dtype = tf.int16))
    else:
      labels = tf.cast(stacked[..., num_channels:], dtype = tf.int16)
    # inputs = rotated[..., :num_channels]
    # labels = rotated[..., num_channels:]

    # images = images/tf.math.max(images)

    return stacked[..., :num_channels], labels

############ generate shadow



def generate_shadow_coordinates(imshape, no_of_shadows=1):
  vertices_list=[]
  for index in range(no_of_shadows):
    vertex=[]
    for dimensions in range(np.random.randint(3,6)):
      ## Dimensionality of the shadow polygon
      vertex.append(( imshape[1]*np.random.uniform(),imshape[0]*np.random.uniform()))
      vertices = np.array([vertex], dtype=np.int32)
      ## single shadow vertices
      vertices_list.append(vertices)
  return vertices_list ## List of shadow vertices


def add_shadow(image,no_of_shadows=1):
  # image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
  ## Conversion to HLS
  image_new = np.zeros_like(image)
  # mask = cv2.UMat(mask)
  imshape = image.shape
  mask = np.zeros(image.shape[:2])

  vertices_list= generate_shadow_coordinates(imshape, no_of_shadows)
  # print(mask.shape)
  shadow_value = 0.4
  # # print(vertices_list)
  # #3 getting list of shadow verticesa
  for vertices in vertices_list:
    mask = poly_to_mask(image.shape[:2], contours = vertices).astype(int)
  mask = np.where(mask, shadow_value, 1.)
  #   poly_to_mask(mask)
  # plt.imshow(mask)
  # plt.show()
    ## adding all shadow polygons on empty mask, single 255 denotes only red channel
  # print(mask.shape)
  mask = tf.convert_to_tensor(mask, dtype= tf.float32)

  # brightness = (0.21 × R) + (0.72 × G) + (0.07 × B)
  image_new[:,:,0] = tf.math.multiply(image[:,:,0], mask) * 1.
  image_new[:,:,1] = tf.math.multiply(image[:,:,1], mask) * 1.
  image_new[:,:,2] = tf.math.multiply(image[:,:,2], mask) * 1.
  # plt.imshow(image[:,:,0], vmin = 0., vmax = 1.)
  # plt.show()
  # image_new[...,1:] = image[..., 1:]
  # image_new[:,:,1][mask] = image[:,:,1][mask]*0.2
  ## if red channel is hot, image's "Lightness" channel's brightness is lowered
  # image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB)
    ## Conversion to RGB
  return tf.convert_to_tensor(image_new)
  # return image


# @tf.function
# def aug_shadow(image):
#   # image = (image*255).numpy().astype(int)
#   # image = cv2.convertScaleAbs(image, alpha=1.0, beta=0)
#   image = add_shadow(image, 3)
#   return image

class RandomShade(layers.Layer):
  def __init__(self, n_shadow, **kwargs):
    super().__init__(**kwargs)
    # self.n_shadow = n_shadow
    self.aug_shadow = partial(add_shadow, no_of_shadows = n_shadow)
  def call( self, image, label ):
    new_image = tf.py_function( self.aug_shadow,
                        [image],
                        'float32',
                        # stateful=False,
                        name='cvOpt')
    new_image = K.stop_gradient( new_image ) # explicitly set no grad
    new_image.set_shape(image.shape) # explicitly set output shape
    return new_image, label

def add_dilation(label, kernel_size, iterations = 1):
  return cv2.dilate(label, np.ones((kernel_size, kernel_size)), iterations = iterations)
class Dilation(layers.Layer):
  def __init__(self, kernel_size = 9, iterations = 1, **kwargs):
      super().__init__(**kwargs)
      # self.n_shadow = n_shadow
      self.kernel_size = kernel_size
      self.dilater = partial(add_dilation, kernel_size = kernel_size, iterations = iterations)
  def call( self, label ):
    dilated_label = tf.py_function( self.dilater,
                        [label],
                        'float32',
                        # stateful=False,
                        name='cvOpt')
    dilated_label = K.stop_gradient( dilated_label ) # explicitly set no grad
    dilated_label.set_shape(label.shape) # explicitly set output shape
    return dilated_label

def random_saturation(image, factor):
  return tf.image.random_saturation(
    image, lower = factor[0], upper = factor[1]
  )
  

class RandomShade(layers.Layer):
  def __init__(self, n_shadow):
    super().__init__()
    # self.n_shadow = n_shadow
    self.aug_shadow = partial(add_shadow, no_of_shadows = n_shadow)
  def call( self, image):
    new_image = tf.py_function( self.aug_shadow,
                        [image],
                        'float32',
                        # stateful=False,
                        name='cvOpt')
    new_image = K.stop_gradient( new_image ) # explicitly set no grad
    new_image.set_shape(image.shape) # explicitly set output shape
    return new_image


  # def compute_output_shape( self, sin ) :
  #   return ( sin[0], 66, 200, sin[-1] )

class RandomSaturation(layers.Layer):
  def __init__(self, factor = []):
    super().__init__()
    # self.n_shadow = n_shadow
    self.aug_sat = partial(random_saturation, factor = factor)
  def call( self, image):
    # image = tfm.vision.preprocess_ops.color_jitter(
    #   image, brightness = 0.4, contrast = 0.4, saturation = 0.3
    # )
    new_image = tf.py_function( self.aug_sat,
                        [image],
                        'float32',
                        # stateful=False,
                        name='cvOpt')
    new_image = K.stop_gradient( new_image ) # explicitly set no grad
    new_image.set_shape(image.shape) # explicitly set output shape
    return new_image

############ generate shadow



