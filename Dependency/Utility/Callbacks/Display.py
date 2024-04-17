import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from Plot.BasicPlot import plotI, plotM

def display(display_list, show = True):
  figure = plt.figure(figsize=(25, 15))
  title = ['Input Image', 'True Mask', 'Predicted Mask in train', 'Predicted Mask in test']
  image_, mask_, pred_mask_train = display_list
  plt.subplot(1, 3, 1)
  plt.title(title[0])

  plotI(image_)

  plt.subplot(1, 3, 2)
  plt.title(title[1])
  plotM(mask_)

  plt.subplot(1, 3, 3)
  plt.title(title[2])
  plotM(pred_mask_train)

  # print(f"loss train: {loss(mask_, pred_mask_train)}")
  if show:
    plt.show()

  return figure

def show_predictions(model, batch_train, batch_test, num=1):
  if batch_test:

    # loss_sum = 0
    plt.figure(figsize=(25, 15))
    idx = np.random.randint(0, 3)
    for image_, mask_ in batch_train.take(num):
      pred_mask_train = model.predict(np.array([image_[idx]]))[0]

      display([image_[idx], mask_[idx], pred_mask_train])
      # display([image[0], mask[0], pred_mask_test[0]])
      # loss_sum = loss_sum + s

    for image_, mask_ in batch_test.take(num):

      pred_mask_test = model.predict(np.array([image_[idx]]))[0]
      # print(pred_mask.shape)
      display([image_[idx], mask_[idx],  pred_mask_test])

class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    # clear_output(wait=True)
    show_predictions()
    print ('\nSample Prediction after epoch {}\n'.format(epoch+1))
