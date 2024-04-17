import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
# import keras

class BasnetLoss(tf.keras.losses.Loss):
    """BASNet hybrid loss."""

    def __init__(self, from_logits = True, class_weights = None, gamma = 2, **kwargs):
        super().__init__(name="basnet_loss", **kwargs)
        self.smooth = 1.0e-9
        self.from_logits = from_logits
        self.class_weights = class_weights
        # Binary Cross Entropy loss.
        if class_weights is None:
          self.cross_entropy_loss = tf.keras.losses.CategoricalFocalCrossentropy(gamma = 2, from_logits= from_logits)
        else:
          self.cross_entropy_loss = tf.keras.losses.CategoricalFocalCrossentropy(alpha = class_weights, gamma = 2, from_logits= from_logits)
       
        # Structural Similarity Index value.
        self.ssim_value = tf.image.ssim
        #  Jaccard / IoU loss.
        self.iou_value = self.calculate_iou

    def calculate_iou(
        self,
        y_true,
        y_pred,
    ):
        """Calculate intersection over union (IoU) between images."""
        intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
        union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3])
        union = union - intersection
        return K.mean(
            (intersection + self.smooth) / (union + self.smooth), axis=0
        )

    def call(self, y_true, y_pred):

        cross_entropy_loss = self.cross_entropy_loss(y_true, y_pred)

        if self.from_logits:
          y_pred = tf.math.softmax(y_pred, axis = -1)

        y_true = tf.cast(y_true, dtype = tf.float32)

        ssim_value = self.ssim_value(y_true, y_pred, max_val=1)
        ssim_loss = K.mean(1 - ssim_value + self.smooth, axis=0)

        iou_value = self.iou_value(y_true, y_pred)
        iou_loss = 1 - iou_value

        # Add all three losses.
        return cross_entropy_loss + ssim_loss + iou_loss

class Multi_stage_loss(tf.keras.losses.Loss):
  """ Multi_stage_loss """
  def __init__(self, loss, n_stage = 7, stage_weights = np.array([4] + [1] * 6) / 10., class_weights = None, **kwargs):
    super().__init__(name="Multi_stage_loss", **kwargs)
    self.loss = loss
    self.n_stage = n_stage

    self.stage_weights = stage_weights

  def call(self, y_true, y_pred):
    """

    """

    combined_loss = self.loss(y_true, y_pred[:, 0, ...]) * self.stage_weights[0]
    for idx in range(1, self.n_stage):
      combined_loss += self.loss(y_true, y_pred[: , idx, ...]) * self.stage_weights[idx]
      
    return combined_loss

class Center_wrap_loss(tf.keras.losses.Loss):
  """ wrap a loss with y_pred cropped to the center """
  def __init__(self, loss, offset, **kwargs):
    super().__init__(name="Center_wrap_loss", **kwargs)
    self.loss = loss
    self.offset = offset

  def call(self, y_true, y_pred):
    """

    """
    y_pred_center = y_pred[..., self.offset: -self.offset, self.offset: -self.offset, :]
    
    return self.loss(y_true, y_pred_center)


class Stacked_loss(tf.keras.losses.Loss):
  """ Multi_stage_loss """
  def __init__(self, loss, weights, **kwargs):
    super().__init__(name="Stacked_binary_loss", **kwargs)
    self.loss = loss
    self.weights = weights

  def call(self, y_true, y_pred):
    """

    """
    combined_loss = self.loss(y_true, y_pred[..., 0]) * self.weights[0]
    for idx in range(1, len(self.weights)):
      combined_loss += self.loss(y_true, y_pred[..., idx]) * self.weights[idx]
      
    return combined_loss

# prompt: create a custom metrics class in tensorflow
