import numpy as np
import matplotlib.pyplot as plt
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

    filtered = median(gray, disk(5))

    canyny_edge = canny(filtered, sigma = 1.5)
    hough_radii = np.arange(20, 35, 2)
    hough_res = hough_circle(canyny_edge, hough_radii)



    plt.imshow(canyny_edge)
    plt.show()



if __name__ == '__main__':
    hough_segmentation(
        '/home/monika/Desktop/project/app/Nanoparticles_app/images/AuNP20nm_004.jpg',
        100)