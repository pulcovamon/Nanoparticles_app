import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.filters import median
from skimage.io import imread
from skimage.transform import (
    rescale, hough_circle, hough_circle_peaks
)
from skimage.feature import canny
from skimage.color import rgb2gray
from skimage.morphology import disk


def hough_segmentation(file_name, scale):
    """Function for image sementation uusing hough elipse transform

    Args:
        file_name (path): path to image

    Returns:

    """

    img_raw = imread(file_name)
    gray = rgb2gray(img_raw)
    gray = rescale(gray, 0.5)
    img_filtered = median(
                    gray, disk(5))

    canny_edge = canny(img_filtered, sigma = 2)

    hough_radii = np.arange(20, 100, 2)
    hough_res = hough_circle(canny_edge, hough_radii)

    accums, x, y, r = hough_circle_peaks(hough_res, hough_radii,
            min_xdistance=10, min_ydistance=10)
    x = np.uint16(np.around(x)) 
    y = np.uint16(np.around(y))
    r = np.uint16(np.around(r))

    circles = [(x[i], y[i], r[i]) for i in range(len(x)) if accums[i] > 0.3]

    img = np.zeros((gray.shape[0], gray.shape[1], 3))
    img[:, :, 0] = gray
    img[:, :, 1] = gray
    img[:, :, 2] = gray

    for x, y, r in circles:
        cv2.circle(img, (x, y), r, (0, 255, 0), 1)
        cv2.circle(img, (x, y), 1, (0, 255, 0), 2)

    return img, circles, 0.5



def calcultaions(circles, pixel_size):
    radius = [r*pixel_size for x, y, r in circles]
    area = [r*pixel_size*np.pi**2 for x, y, r in circles]

    mean_radius = np.mean(radius)
    mean_area = np.mean(area)

    return mean_radius, mean_area


def plotting(img):

    plt.imshow(img)
    plt.show()



if __name__ == '__main__':
    img, circles, pixel_size = hough_segmentation(
        '/home/monika/Desktop/project/Nanoparticles_app/images/AuNP20nm_004.jpg',
        100)

    radius, area = calcultaions(circles, pixel_size)

    plotting(img)