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

    sobel_edges = sobel(img_filtered)

    thresh = threshold_otsu(sobel_edges)
    binary_edges = sobel_edges > thresh

    binary_edges = binary_closing(binary_edges)

    region_fill = ndi.binary_fill_holes(binary_edges)
    erode = erosion(region_fill, square(width = 5))

    #anyny_edge = canny(filtered, sigma = 1.5)
    hough_radii = np.arange(20, 35, 2)
    hough_res = hough_circle(erode, hough_radii)

    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                           total_num_peaks=3)


    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
    image = gray2rgb(img_raw)
    for center_y, center_x, radius in zip(cy, cx, radii):
        circy, circx = circle_perimeter(center_y, center_x, radius,
                                        shape=image.shape)
        image[circy, circx] = (220, 20, 20)

    ax.imshow(image, cmap=plt.cm.gray)
    plt.show()



if __name__ == '__main__':
    hough_segmentation(
        '/home/monika/Desktop/project/Nanoparticles_app/images/AuNP20nm_004.jpg',
        100)