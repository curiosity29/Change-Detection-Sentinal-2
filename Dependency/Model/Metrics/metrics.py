import tensorflow as tf


class U2Net_metric(tf.keras.metrics.Metric):

  def __init__(self, name='accuracy', base_metric = tf.keras.metrics.BinaryAccuracy(), **kwargs):
    super().__init__(name=name, **kwargs)
    # self.total = self.add_weight(name='total', initializer='zeros')
    # self.count = self.add_weight(name='count', initializer='zeros')
    self.metric = base_metric

  def update_state(self, y_true, y_pred, sample_weight=None):
    # values = tf.cast(y_pred, tf.float32)
    # y_max = y_pred[:, 0, :, :, :]
    self.metric.update_state(y_true, y_pred[:, 0, :, :, :], sample_weight)
    # self.metric.total.assign_add(values)
    # self.metric.count.assign_add(1)

  def result(self):
    return self.metric.result()

  def reset_states(self):
    self.metric.reset_states()


class Stacked_metric(tf.keras.metrics.Metric):

  def __init__(self, name='accuracy', index = 0, base_metric = tf.keras.metrics.BinaryAccuracy(), **kwargs):
    super().__init__(name=name, **kwargs)
    # self.total = self.add_weight(name='total', initializer='zeros')
    # self.count = self.add_weight(name='count', initializer='zeros')
    self.metric = base_metric
    self.index = index

  def update_state(self, y_true, y_pred, sample_weight=None):
    # values = tf.cast(y_pred, tf.float32)
    # y_max = y_pred[:, 0, :, :, :]
    self.metric.update_state(y_true, y_pred[..., self.index], sample_weight)
    # self.metric.total.assign_add(values)
    # self.metric.count.assign_add(1)

  def result(self):
    return self.metric.result()

  def reset_states(self):
    self.metric.reset_states()