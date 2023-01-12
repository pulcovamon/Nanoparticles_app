from skimage.io import imread, imsave
import numpy as np
import matplotlib.pyplot as plt

img = imread(r'icons/image_icon.png')

#img = img[:, :, 3]

img[:, :, 0:3] = 255 - img[:, :, 0:3]

plt.imsave('icons/image_icon.png', img)