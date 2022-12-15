import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.filters import median, threshold_otsu, sobel
from skimage.io import imread
from skimage.transform import (
    rescale, hough_circle, hough_circle_peaks
)
from skimage.feature import canny
from skimage.color import rgb2gray, gray2rgb
from skimage.morphology import (
    disk, binary_closing, erosion, square
)
from skimage.util import invert
from skimage.draw import circle_perimeter

def hough_segmentation(file_name, scale):
    """Function for image sementation uusing hough elipse transform

    Args:
        file_name (path): path to image

    Returns:

    """

    img_raw = imread(file_name)
    gray = rgb2gray(img_raw)

    img_raw = rgb2gray(img_raw)
    img_raw = rescale(img_raw, 0.5)
    img_inverted = invert(img_raw)

    img_filtered = median(
                    img_raw, disk(5))

    canny_edge = canny(img_filtered, sigma = 2)

    #anyny_edge = canny(filtered, sigma = 1.5)
    hough_radii = np.arange(20, 100, 2)
    hough_res = hough_circle(canny_edge, hough_radii)

    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks = 100)


    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
    canny_edge = canny_edge * 255
    canny_edge = canny_edge.astype(np.uint8)
    image = gray2rgb(canny_edge)
    for center_y, center_x, radius in zip(cy, cx, radii):
        circy, circx = circle_perimeter(center_y, center_x, radius,
                                        shape=image.shape)
        image[circy, circx] = (255, 0, 0)

    ax.imshow(image)
    plt.show()



if __name__ == '__main__':
    hough_segmentation(
        '/home/monika/Desktop/project/app/Nanoparticles_app/images/AuNP20nm_004.jpg',
        100)