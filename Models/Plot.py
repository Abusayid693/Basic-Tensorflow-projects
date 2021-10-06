import numpy as np
import matplotlib.ticker as mticker
from matplotlib import pyplot as plt

def plot_metrics( train_m , val_m ,title):
  plt.title(title)
  plt.ylim(0,1.0)
  plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
  plt.plot(train_m,color="blue")
  plt.plot(val_m,color="red")


def display_image( image_to_plot, n, y_pred_label, y_real_label):
  plt.figure(figsize=(17,3)) 
  display_strings = [str(i)+"\n\n"+str(j) for i,j in zip(y_pred_label,y_real_label)]
  plt.xticks([28*x+14 for x in range(n)],display_strings )
  image = np.reshape(image_to_plot , [n,28,28])
  image = np.swapaxes( image , 0, 1)
  image = np.reshape( image, [28, 28 * n])
  plt.imshow(image)

plot_metrics(epoch_train_losses,epoch_val_losses, "loss")


def Visualize_metrice():
  # Corresponding labels for Fashion MNST datasets
  classes_name = [ "T-shirt/top","Trouser/pants", "Pullover shirt","Dress","Coat","Sandal", "Shirt", "Sneaker","Bag","Ankle boot" ]
  x_batches, y_pred_batches, y_real_batches=[],[],[]

  for x,y in dev:
    y_pred = model(x)
    y_pred_batches = y_pred.numpy()
    y_real_batches = y.numpy()
    x_batches = x.numpy()

  indexes = np.random.choice(len(y_pred_batches), size=10)
  image_to_plot = x_batches[indexes]
  y_pred_to_plot = y_pred_batches[indexes]
  y_true_to_plot = y_real_batches[indexes]
  y_pred_label = [ classes_name[np.argmax(j)] for j in y_pred_to_plot]
  y_real_label = [ classes_name[i] for i in y_true_to_plot]
  display_image( image_to_plot, 10, y_pred_label, y_real_label)

Visualize_metrice()