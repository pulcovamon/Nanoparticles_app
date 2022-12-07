import numpy as np
from skimage import data, restoration
from matplotlib import pyplot as plt

img = data.coins()
background = restoration.rolling_ball(img)
substracted = img - background

plt.subplot(2,2, 1)
plt.imshow(img, cmap = 'gray')

plt.subplot(2, 2, 2)
plt.imshow(background, cmap = 'gray')

plt.subplot(2, 2, 3)
plt.imshow(substracted, cmap = 'gray')
plt.show()