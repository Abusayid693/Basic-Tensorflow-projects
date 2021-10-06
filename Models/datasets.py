"""
Tensorflow fashion_mnist datasets 
   @tfds :  
   TensorFlow Datasets is a collection of datasets ready to use, with TensorFlow 
   or other Python ML frameworks, such as Jax. All datasets are exposed as tf.data.Datasets ,
   enabling easy-to-use and high-performance input pipelines. To get started see the guide and our 
   list of datasets.
"""

import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

def load_datasets():
  train_data , info = tfds.load("fashion_mnist", split="train", with_info=True)
  test_data = tfds.load("fashion_mnist", split="test")
  # Fashion_mnist contains 10 labels 
  Num_of_training_data = tf.data.experimental.cardinality(train_data).numpy()
  print(Num_of_training_data)
  print(train_data)
  return train_data, test_data

"""
Normalizing the datasets 
"""

def Normalize_datasets(data):
  image = data["image"]
  # Rescaling image into a vector
  image = tf.reshape(image, [-1,])
  image = tf.cast(image,'float32')
  image = image/255.0
  return image , data["label"]  

"""
Normalizing the datasets and forming batches from original
size of 60,000 training examples and 10,000
test examples
"""

def preprocess_datasets(batch_size):
  # Getting datasets from load_datasets function
  train_data, test_data = load_datasets()
  train_data = train_data.map(Normalize_datasets)
  test_data = test_data.map(Normalize_datasets)
  # Random shuffling with default buffer size 
  train = train_data.shuffle(buffer_size=1024).batch(batch_size)
  dev = test_data.batch(batch_size)
  return train, dev


