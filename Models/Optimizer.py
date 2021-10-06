"""
Defining deep neural network model using keras layers
 Input : 784 neurons
 1st Deep layer : 64 neurons, relu activation
 2nd Deep layer : 64 neurons, relu activation
 Output layer : 10 neurons, softmax activation
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
import numpy as np

def base_model():
  inputs = tf.keras.Input(shape = (784,), name="clothing")
  x = tf.keras.layers.Dense(64, activation="relu", name="dense_1")(inputs)
  x = tf.keras.layers.Dense(64, activation="relu", name="dense_2")(x)
  outputs = tf.keras.layers.Dense(10, activation="softmax" , name="predictions")(x)
  model = tf.keras.Model(inputs=inputs,outputs=outputs)
  return model

"""
Adam optimization is a stochastic gradient descent method that is based on 
adaptive estimation of first-order and second-order moments. We used SparseCategoricalCrossentropy
since our labels are not in one-hot form.
"""

def optimizer_and_loss(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name='Adam'
    ):
  optimizer = tf.keras.optimizers.Adam(
      learning_rate,
      beta_1,
      beta_2,
      epsilon,
      amsgrad,
      name)
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
  return optimizer, loss_object
  