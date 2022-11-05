import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from skimage import filters, morphology
from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

img_raw = imread('AuNP20nm_004.jpg')
img_raw = img_raw[0:-100, :]
img_raw = rgb2gray(img_raw)

img = filters.median(img_raw, np.ones((11, 11)))

threshold = threshold_otsu(img)
img_binary = img > threshold
img_binary = ~img_binary

kernel = morphology.square(width = 5)
img_erode = morphology.erosion(img_binary, kernel)
img_dilate = morphology.dilation(img_erode, kernel)

distance = ndi.distance_transform_edt(img_dilate)
img_mask = distance > 0.5 * np.amax(distance)
markers, _ = ndi.label(img_mask)
img_segment = watershed(-distance, markers, mask = img_binary)

plt.subplot(2, 2, 1)
plt.imshow(img_binary, cmap = 'gray')
plt.subplot(2, 2, 2)
plt.imshow(img_erode, cmap = 'gray')
plt.subplot(2, 2, 3)
plt.imshow(img_dilate, cmap = 'gray')
plt.subplot(2, 2, 4)
plt.imshow(img_segment, cmap = 'tab20')
plt.show()