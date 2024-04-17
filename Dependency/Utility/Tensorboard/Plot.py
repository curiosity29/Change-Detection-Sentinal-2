import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import io
import itertools
from sklearn.metrics import confusion_matrix

def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image

def plot_confusion_matrix(cm, class_names):
  """
  Returns a matplotlib figure containing the plotted confusion matrix.

  Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
  """
  figure = plt.figure(figsize=(8, 8))
  plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
  plt.title("Confusion matrix")
  plt.colorbar()
  tick_marks = np.arange(len(class_names))
  plt.xticks(tick_marks, class_names, rotation=45)
  plt.yticks(tick_marks, class_names)

  # Compute the labels from the normalized confusion matrix.
  labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

  # Use white text if squares are dark; otherwise black.
  threshold = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    color = "white" if cm[i, j] > threshold else "black"
    plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  return figure

def log_predict(epoch, logs, logdir, model, batch_test, freq = 1, **ignore):
  file_writer_lb = tf.summary.create_file_writer(logdir + '/predict_label')
  for sample_image, sample_label in batch_test.take(1):
    pass
  test_pred_raw = model.predict(np.array([sample_image[0]]))
  test_pred = tf.cast(test_pred_raw, dtype = tf.int16)

  figure = plt.figure(figsize=(16, 16))
  plt.subplot(131)
  plt.imshow(sample_image[0])
  plt.subplot(132)
  plt.imshow(sample_label[0])
  plt.subplot(133)
  plt.imshow(test_pred[0], vmin = 0., vmax = 1.)
  with file_writer_lb.as_default():
    tf.summary.image("predict", plot_to_image(figure), step=epoch)
  

def log_confusion_matrix(epoch, logs, logdir, model, batch_test, class_names = None, freq = 5, output_adapter = lambda x: x):
  file_writer_cm = tf.summary.create_file_writer(logdir + '/confusion_matrix')
  # Use the model to predict the values from the validation dataset.
  if epoch % freq != 0:
    return
  for sample_image, sample_label in batch_test.take(1):
    pass
  # num = 10
  # rand_idx = np.random.randint(low = 0, high = n-num-1)
  # test_pred_raw = model.predict(images_data_test[num:num+10])
  test_pred = model.predict(sample_image)
  # test_pred = np.argmax(test_pred_raw, axis=-1)
  test_pred = output_adapter(test_pred)

  # Calculate the confusion matrix.
  cm = confusion_matrix(np.argmax(sample_label, axis = -1).reshape(-1), test_pred.reshape(-1))
  # Log the confusion matrix as an image summary.
  if class_names is None:
    figure = plot_confusion_matrix(cm)
  else:
    figure = plot_confusion_matrix(cm, class_names=class_names)
  cm_image = plot_to_image(figure)
  # random_idx = np.random.randint(0, n)
  # display_list = [images_data_test[random_idx], labels_data_test[random_idx], model.predict(np.array([images_data_test[random_idx]]))[0]]

  # Log the confusion matrix as an image summary.
  with file_writer_cm.as_default():
    tf.summary.image("confusion_matrix", cm_image, step=epoch)
