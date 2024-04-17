import tensorflow as tf
import glob, os
def checkpoint_name_auto(folder, problemName = "segmentation1", modelName = "modelName", date = "today", version = "v1", **ignore):
  checkpoint_save_best_loss = f"{folder}{problemName}_{modelName}_{date}_{version}_loss.weights.h5"
  checkpoint_save_best_accuracy = f"{folder}{problemName}_{modelName}_{date}_{version}_accuracy.weights.h5"
  checkpoint_save_last = f"{folder}{problemName}_{modelName}_{date}_{version}_last.weights.h5"

  return checkpoint_save_best_loss, checkpoint_save_best_accuracy, checkpoint_save_last


def save_callbacks_3(folder, problemName = "segmentation1", modelName = "modelName", date = "today", version = "v1", save_freq = 80):
  args = locals()
  checkpoint_save_best_loss, checkpoint_save_best_accuracy, checkpoint_save_last =\
  checkpoint_name_auto(**args)

  loss_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_best_loss,
                                                  monitor = "val_loss",
                                                  mode = 'min',
                                                  save_best_only = True,
                                                  save_weights_only = True,
                                                  verbose=1)

  accuracy_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_best_accuracy,
                                                  monitor = "accuracy",
                                                  mode = 'max',
                                                  save_best_only = True,
                                                  save_weights_only = True,
                                                  verbose=1)

  crash_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_last,
                                                  monitor = "val_loss",
                                                  mode = 'min',
                                                  save_freq = save_freq,
                                                  save_weights_only = True,
                                                  verbose=1)

  return [loss_callback, accuracy_callback, crash_callback], [checkpoint_save_best_loss, checkpoint_save_best_accuracy, checkpoint_save_last]