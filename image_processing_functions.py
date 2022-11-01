from gettext import find
from operator import index
import cv2
from cv2 import threshold
import numpy as np
# ()


def loadImg():

    '''
    Function for loading image, get the ROI and display the croped image
    Parameters:
    ------
    '''

    img = cv2.imread('AuNP50nm_008.jpg', cv2.IMREAD_GRAYSCALE)
    #img = img[my_index[0][0]:my_index[-1][0], my_index]
    return img


def filterImg(img):

    '''
    Function for bluring the image and substract the background
    Parameters:
    img - np.ndarray uint8 grayscale
    '''

    bck = cv2.GaussianBlur(img, (169, 169), 0)
    img = cv2.medianBlur(img, 5)
    #img = img / bck
    return img

def binaryImg(img):

    '''
    Function for transform the grayscale image to binary image and apply closing and opening
    Parameters:
    img - np.ndarray uint8 grayscale
    '''

    ret, img = threshold(img, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    my_kernel = (5, 5)
    img = cv2.dilate(img, kernel = my_kernel)
    img = cv2.erode(img, kernel = my_kernel)
    return img