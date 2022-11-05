import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from skimage import filters, morphology
from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from skimage.segmentation import watershed
from skimage.feature import peak_local_max


def loading_img(img_path):
    """Function for loading image from given file and croping it

    Args:
        img_path (str): path to image file

    Returns:
        numpy.ndarray: RGB image
    """

    img_raw = imread(img_path)
    img_raw = img_raw[0:-100, :]

    return img_raw



def filtering_img(img_raw):
    """Function for bluring image and edge detection

    Args:
        img_raw (numpy.ndarray): RGB image

    Returns:
        numpy.ndarray: grayscale image

    """

    img_raw = rgb2gray(img_raw)

    img_median = filters.median(img_raw, np.ones((11, 11)))
    sobel_edges = filters.sobel(img_median)

    return sobel_edges



def edges_operations(sobel_edges):
    """Function for transforming edges image to binary image using thresholding and morfological operations

    Args:
        sobel_edges (numpy.ndarray): grayscale image of edges

    Returns:
        numpy.ndarray: binary image
    """

    thresh = threshold_otsu(sobel_edges)
    binary_edges = sobel_edges > thresh

    binary_edges = morphology.binary_closing(binary_edges)

    region_fill = ndi.binary_fill_holes(binary_edges)
    erode = morphology.erosion(region_fill, morphology.square(width = 5))

    return region_fill, erode



def watershed_transform(erode, region_fill):
    """Function for labeling of image using calculated distance map and markers for watershed transform

    Args:
        erode (numpy.ndarray): binary image
        region_fill (numpy.ndarray): binary image

    Returns:
        numpy.ndarray: labeled image
    """

    distance = ndi.distance_transform_edt(erode)

    coords = peak_local_max(distance, footprint = np.ones((5, 5)), min_distance = 30)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True

    markers, _ = ndi.label(mask)

    labels = watershed(-distance, markers, mask = region_fill)

    return labels


def ploting_img(img_raw, sobel_edges, region_fill, labels):
    """Function for plotting images from segmentation proces

    Args:
        img_raw (numpy.ndarray): RGB image
        sobel_edges (numpy.ndarray): grayscale image of edges
        region_fill (_numpy.ndarray): binary image
        labels (numpy.ndarray): labeled image
    """

    plt.subplot(2, 2, 1)
    plt.imshow(img_raw)
    plt.title('Raw image')

    plt.subplot(2, 2, 2)
    plt.imshow(sobel_edges, cmap = 'gray')
    plt.title('Edges detected using Sobel operator')

    plt.subplot(2, 2, 3)
    plt.imshow(region_fill, cmap = 'gray')
    plt.title('Binary image')

    plt.subplot(2, 2, 4)
    plt.imshow(labels, cmap = 'tab20')
    plt.title('Labeled image')

    plt.show()


if __name__ == '__main__':

    img_path = 'AuNP20nm_004.jpg'

    img_raw = loading_img(img_path)

    sobel_edges = filtering_img(img_raw)

    region_fill, erode = edges_operations(sobel_edges)

    labels = watershed_transform(erode, region_fill)

    ploting_img(img_raw, sobel_edges, region_fill, labels)