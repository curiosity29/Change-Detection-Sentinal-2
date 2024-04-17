import tensorflow as tf
from tensorflow.keras import layers

######################################
class CoBaRe(layers.Layer):
  """
    convolution, batch normalization, relu
  """
  def __init__(self, filters = 64, kernel_size = 3,
               padding = "same", dilation_rate = 1, use_bias = True, transpose = False,
               convArgs = {},
               **kwargs):
    super(CoBaRe, self).__init__(**kwargs)
    if transpose:
      self.conv = layers.Conv2DTranspose(
        filters = filters, kernel_size = kernel_size,
        padding = padding, dilation_rate = dilation_rate, use_bias = use_bias, **convArgs)
    else:
      self.conv = layers.Conv2D(
        filters = filters, kernel_size = kernel_size,
        padding = padding, dilation_rate = dilation_rate, use_bias = use_bias, **convArgs)
    self.batch = layers.BatchNormalization(axis = -1)
    # self.relu = layers.Activation('relu')

  def call(self, x):
    return tf.nn.relu(self.batch(self.conv(x)))

  def get_config(self):
    config = super().get_config()
    config.update(
      {
        "conv": self.conv,
        "batch": self.batch
      }
    )
    return config

  @classmethod
  def from_config(cls, config):
    config["conv"] = tf.keras.layers.deserialize(config["conv"])
    config["batch"] = tf.keras.layers.deserialize(config["batch"])
        
    return cls(**config)

######################################

class CoSigUp(layers.Layer):
  def __init__(self, up_size = 2, filters = 1, kernel_size = 3,
               padding = "same", dilation_rate = 1, use_bias = True, transpose = False,
               **kwargs):
    super(CoSigUp, self).__init__(**kwargs)
    if transpose:
      self.conv = layers.Conv2DTranspose(
        filters = filters, kernel_size = kernel_size,
        padding = padding, dilation_rate = dilation_rate, use_bias = use_bias)
    else:
      self.conv = layers.Conv2D(
        filters = filters, kernel_size = kernel_size,
        padding = padding, dilation_rate = dilation_rate, use_bias = use_bias)
    self.upper = layers.UpSampling2D(up_size)
    # self.batch = layers.Up(axis = -1)
    # self.relu = layers.Activation('relu')

  def call(self, x):
    x = self.conv(x)
    x = tf.nn.sigmoid(x)
    return self.upper(x)

  def get_config(self):
    config = super().get_config()
    config.update(
      {
        "conv": self.conv,
        "upper": self.upper
      }
    )
    return config

  @classmethod
  def from_config(cls, config):
    config["conv"] = tf.keras.layers.deserialize(config["conv"])
    config["upper"] = tf.keras.layers.deserialize(config["upper"])      
    return cls(**config)
  
######################################

def DilatedSpatialPyramidPooling(dspp_input, filters = 256, drop_out = None, mode = "down", dilations = [[1,1], [3,1], [3,6]], **regularizer):
    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = CoBaRe(kernel_size=1, use_bias=True, filters =filters)(x)
    out_pool = layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
    )(x)
    conv_list = []
    if mode == "down":
      for rate in dilations:
        conv_list.append(CoBaRe(filters = filters, kernel_size=rate[0], dilation_rate=rate[1], **regularizer)(dspp_input))
    else:
      for rate in dilations:
        conv_list.append(CoBaRe(filters = filters, kernel_size=rate[0], dilation_rate=rate[1], transpose = True, **regularizer)(dspp_input))

    x = layers.Concatenate(axis=-1)([out_pool] + conv_list)
    # x = layers.Concatenate(axis=-1)([out_pool, out_1, out_2, out_3, out_4, out_v, out_h])
    if drop_out is not None:
      x = layers.Dropout(rate = drop_out)(x)

    if mode == "down":
      output = CoBaRe(filters = filters, kernel_size=1, **regularizer)(x)
    else:
      output = CoBaRe(filters = filters, kernel_size=1, transpose = True, **regularizer)(x)

    return output

######################################

def UNet_encoder(x, filters):
  """
    Args:
      x: input abtract tensor
      filters: list of filter for each step
  """
  skips = []
  for idx, filter_ in enumerate(filters):
    x = CoBaRe(filters = filter_, name = f"en_{idx+1}_1")(x)
    x = CoBaRe(filters = filter_, name = f"en_{idx+1}_2")(x)
    skips.append(x)
    x = layers.MaxPooling2D(2, name = f"down_{idx+1}")(x)
  return x, skips

######################################

def UNet_decoder(x, filters, skips):
  depth = len(filters)
  for idx, filter_ in enumerate(filters):
    x = layers.UpSampling2D(2, name = f"up_{depth-idx}")(x)
    x = layers.Concatenate(axis = -1, name = f"skip_{depth-idx}")([x, skips.pop()])
    x = CoBaRe(filters = filter_, transpose = True, name = f"de_{depth-idx}_1")(x)
    x = CoBaRe(filters = filter_, transpose = True, name = f"de_{depth-idx}_2")(x)
  return x

######################################

def dilated_UNet_encoder(x, filters):
  skips = []
  for idx, filter_ in enumerate(filters):
    x = DilatedSpatialPyramidPooling(x, filters = filter_)
    x = CoBaRe(filters = filter_, name = f"en_{idx+1}_2", transpose = True)(x)
    skips.append(x)
    x = layers.MaxPooling2D(2, name = f"down_{idx+1}")(x)
  return x, skips

######################################

def mlp(x, hidden_units, dropout_rate = 0.0, **args):
    for units in hidden_units:
        x = layers.Dense(units, **args)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

######################################

