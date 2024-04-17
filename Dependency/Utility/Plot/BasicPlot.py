import matplotlib.pyplot as plt
import tensorflow as tf

def plotI(image):
  plt.imshow(tf.keras.utils.array_to_img(image[..., :3]))
  # plt.show()
def plotM(mask, max_head = True, n_class = 2):
  if max_head:
    mask = tf.math.argmax(mask, axis = -1)
  plt.imshow(mask, vmin=0, vmax = n_class-1)
  # plt.show()

# def plotV(image, cmap = "gray"):
#   plt.imshow(image[..., 4], vmin=0, vmax = n_class - 1, cmap = "gray")
#   # plt.show()

# def plotW(image, cmap = "gray"):
#   plt.imshow(image[..., 5], vmin=0, vmax = n_class - 1, cmap = "gray")
#   # plt.show()

def show2(im1, im2):
  plt.subplot(121)
  plt.imshow(im1)
  plt.subplot(122)
  plt.imshow(im2)

def plotPair(image, mask):
  plt.subplot(121)
  plotI(image)
  plt.subplot(122)
  plotM(mask)