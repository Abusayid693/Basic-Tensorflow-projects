import tensorflow as tf

"""
Defining mertics to keep a track-record of our model accuracy during training processs
"""

def initialize_accuracy_matrics():
  train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
  val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
  return train_acc_metric, val_acc_metric

