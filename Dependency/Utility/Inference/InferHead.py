import tensorflow as tf
from tensorflow.keras import layers

def softmaxHead(model):
  return tf.keras.Model(model.input, tf.nn.softmax(model.output))

def infer_head_max(model):
  """
    argmax with output last axis to dim = 1
  """
  return tf.keras.Model(inputs = model.input,
  outputs = tf.expand_dims(
    tf.math.argmax(model.output, axis = -1), axis = -1
    ))
def infer_head_confident(model, threshold = 0.9):
  """
    create a 0 mask to non confident pixel (prob < 0.9), keep the rest
  """
  input = tf.keras.Input(shape = model.input.get_shape()[1:])
  x = model(input)
  # create class 0
  x = tf.concat([tf.zeros_like(x[..., :1]), x], axis=-1)
  confident = tf.nn.relu(tf.math.sign(x - threshold))
  x = layers.Multiply()([x, confident])

  # output class, if not confident -> output class is 0
  output = tf.math.argmax(x, axis = -1)
  return tf.keras.Model(inputs = input, outputs = output)

def infer_head_confident_level(model):
  return tf.keras.Model(inputs = model.input, outputs = tf.math.reduce_max(model.output, axis = -1))

def infer_head_indecisive(model, threshold = 0.5):
  """
    create a 0 mask to non confident pixel (prob < 0.9), keep the rest
  """
  input = tf.keras.Input(shape = model.input.get_shape()[1:])
  x = model(input)
  # create class 0
  x = tf.concat([tf.zeros_like(x[..., :1]), x], axis=-1)
  confident = tf.nn.relu(tf.math.sign(-x + threshold))
  x = layers.Multiply()([x, confident])

  # output class, if confident -> output class is 0
  output = tf.math.argmin(x, axis = -1)

  return tf.keras.Model(inputs = model.input, outputs = tf.math.argmax(model.output, axis = -1))

def infer_u2net(model):
  inputs = tf.keras.Input(shape = model.input.shape[1:])
  outputs = model(inputs)[:, 0, ...]
  return tf.keras.Model(inputs = inputs, outputs = outputs)
