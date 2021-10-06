import tensorflow as tf
import itertools
from tqdm import tqdm

"""
Defining training loop with tensorflow gradient tape
"""

def apply_gradient(optimizer,model,x,y,loss_object):
  """
  tf.GradientTape allows us to track TensorFlow computations and calculate
   gradients w.r.t. (with respect to) some given variables
        @params:
        :optimizer -- main optimizer function (i.e : )
        :model -- dictionary mapping words to their GloVe vector representation
        :x -- number of words in the vocabulary
        :y -- number of words in the vocabulary
        :loss_object -- number of words in the vocabulary
        @return:
        :decoder_embedding -- pretrained layer Keras instance
  """
  with tf.GradientTape() as tape:
    # Forward propogation
    logits = model(x)
    loss_value = loss_object(y_true = y, y_pred = logits)
  # backward propogation i.e : Gradient computation
  gradients = tape.gradient(loss_value, model.trainable_weights)
  optimizer.apply_gradients(zip(gradients,model.trainable_weights))  
  return logits,loss_value


"""
Defining train method for one epoch in mini batches
"""

def train_data_for_one_epoch(optimizer, train, model, loss_object, train_acc_metric):

 """
  tf.GradientTape allows us to track TensorFlow computations and calculate
   gradients w.r.t. (with respect to) some given variables
        @params:
        :optimizer -- main optimizer function (i.e : )
        :model -- dictionary mapping words to their GloVe vector representation
        :x -- number of words in the vocabulary
        :y -- number of words in the vocabulary
        :loss_object -- number of words in the vocabulary
        @return:
        :decoder_embedding -- pretrained layer Keras instance
  """  
  losses = []

  pbar= tqdm(total=len(list(enumerate(train))), position=0, leave=True, bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt}')

  for step, (x_batch_train, y_batch_train) in enumerate(train):
    logits,loss_value = apply_gradient(optimizer,model,x_batch_train,y_batch_train,loss_object)
    losses.append(loss_value)
    train_acc_metric(y_batch_train,logits)
    pbar.set_description("Training loss for step %s: %.4f" % ( int(step), float(loss_value)))
    pbar.update()
  return losses



"""
Performing corss validation on development set after each epoch 
and calculating cross-accuracy
"""
def perform_validation(model, dev, val_acc_metric ,loss_object):
  losses = []
  for x_val,y_val in dev:
    val_logits = model(x_val)
    val_loss = loss_object(y_true = y_val, y_pred = val_logits)
    losses.append(val_loss)
    val_acc_metric(y_val,val_logits)
  return losses 
 
