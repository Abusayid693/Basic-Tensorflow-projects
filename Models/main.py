import numpy as np

"""
Training loop initialization
"""

def initialize_training():

  train_acc_metric, val_acc_metric = initialize_accuracy_matrics()
  optimizer,loss_object = optimizer_and_loss()
  train, dev = preprocess_datasets( 64)
  # Instance of main tensorflow model
  model = base_model()
  epochs = 10
  epoch_val_losses, epoch_train_losses = [],[]

  for epoch in range(epochs):
    print('Start of epoch %d'% (epoch,)) 
    # Train set evaluation
    losses_train = train_data_for_one_epoch(optimizer,train,model,loss_object, train_acc_metric)
    train_acc = train_acc_metric.result()
    # Dev set evaluation
    losses_val = perform_validation(model, dev , val_acc_metric, loss_object)
    val_acc = val_acc_metric.result()
    # Mean over all batches
    losses_train_mean = np.mean(losses_train)
    losses_dev_mean = np.mean(losses_val)

    epoch_val_losses.append(losses_train_mean)
    epoch_train_losses.append(losses_dev_mean)

    print("For epoch"+str(epoch)+", Train loss is = "+str(losses_train_mean)+", Dev loss is ="+str(losses_dev_mean)+" Train accuracy is ="+str(train_acc.numpy())+", Dev accuracy is "+str(val_acc.numpy()))
    # Resetting evaluation metrics
    train_acc_metric.reset_states()
    val_acc_metric.reset_states()

  return model, epoch_val_losses, epoch_train_losses, dev