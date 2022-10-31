import cv2
import numpy as np

img = cv2.imread('AuNP50nm_008.jpg', cv2.IMREAD_GRAYSCALE)

img = img[700:1300, 900:1300]

bck = cv2.GaussianBlur(img, (169, 169), 0)
img = cv2.medianBlur(img, 5)

#img = img / bck

ret, img2 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cv2.imshow('orig', img)
cv2.imshow('background', bck)
cv2.waitKey(0)