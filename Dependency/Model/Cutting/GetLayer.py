import tensorflow as tf
def cut_model(model, layername, input = True):
    """
      not finished
    """
    
    if input:
      return tf.keras.Model(inputs = model.input, outputs = model.get_layer(layername).output)

    else:
      return tf.keras.Model(inputs = model.get_layer(layername).output, outputs = model.output)

# model = cut_model(model, "boosting_input", input = True)