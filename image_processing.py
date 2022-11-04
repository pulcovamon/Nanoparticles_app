import cv2
import numpy as np
import matplotlib.pyplot as plt

img_raw = cv2.imread('AuNP20nm_004.jpg')

#img_raw = img_raw[700:1300, 900:1300]
img_raw = img_raw[0:-100, :]

img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
bck = cv2.GaussianBlur(img, (169, 169), 0)
img = cv2.medianBlur(img, 5)

#bcg_substractor = cv2.createBackgroundSubtractorMOG2(detectShadows = False)
#fmask = bcg_substractor.apply(img)

ret, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

img = 255-img

frg_kernel = np.ones((9, 9), np.uint8)
bcg_kernel = np.ones((5, 5), np.uint8)
bcg = cv2.dilate(img, bcg_kernel, iterations=1)
frg = cv2.erode(img, frg_kernel, iterations=6)

distance_map = cv2.distanceTransform(img, distanceType=cv2.DIST_C, maskSize=5)
ret, frg = cv2.threshold(frg, 0.7 * distance_map.max(), 255, 0)
frg = np.uint8(frg)
particles = cv2.subtract(bcg, frg)

ret, labels = cv2.connectedComponents(frg)
labels = labels + 1
labels[particles == 255] = 0

labels = cv2.watershed(img_raw, labels)

plt.subplot(2, 2, 1)
plt.imshow(labels, cmap='tab20')

plt.subplot(2, 2, 2)
plt.imshow(frg)

plt.subplot(2, 2, 3)
plt.imshow(distance_map)

plt.show()