import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from pathlib import Path
import math

from skimage import filters, morphology, util
from skimage.io import imread, imsave
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray, label2rgb
from skimage.segmentation import watershed
from skimage.feature import (
        peak_local_max
            )
from skimage.transform import rescale
from skimage.restoration import rolling_ball
from skimage.morphology import disk, white_tophat
from skimage.util import img_as_ubyte
from skimage.measure import label, regionprops
from skimage.draw import circle_perimeter, ellipse_perimeter
import cv2
# () /


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
    img_raw = rescale(img_raw, 0.5)

    img_filtered = filters.median(
                    img_raw, morphology.disk(5))

    return img_raw



def edges_operations(img_filtered):
    """Function for transforming edges image to binary image using thresholding and morfological operations

    Args:
        sobel_edges (numpy.ndarray): grayscale image of edges

    Returns:
        numpy.ndarray: binary image
    """

    thresh = threshold_otsu(img_filtered)
    binary = img_filtered < thresh

    return binary



def watershed_transform(binary):
    """Function for labeling of image using calculated distance map and markers for watershed transform

    Args:
        erode (numpy.ndarray): binary image
        region_fill (numpy.ndarray): binary image

    Returns:
        numpy.ndarray: labeled image
    """

    distance = ndi.distance_transform_edt(binary)

    coords = peak_local_max(distance, footprint = np.ones((3, 3)), min_distance = 20)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True

    markers, _ = ndi.label(mask)

    labels = watershed(-distance, markers, mask = binary)

    return labels, distance


def ploting_img(img_median, binary, distance, labels):
    """Function for plotting images from segmentation proces

    Args:
        img_raw (numpy.ndarray): RGB image
        sobel_edges (numpy.ndarray): grayscale image of edges
        region_fill (_numpy.ndarray): binary image
        labels (numpy.ndarray): labeled image
    """

    plt.subplot(2, 2, 1)
    plt.imshow(img_median,cmap = 'gray')
    plt.title('Raw image')

    plt.subplot(2, 2, 2)
    plt.imshow(binary, cmap = 'gray')
    plt.title('Binary')

    plt.subplot(2, 2, 3)
    plt.imshow(distance, cmap = 'gray')
    plt.title('Distance')

    plt.subplot(2, 2, 4)
    plt.imshow(labels, cmap = 'tab20')
    plt.title('Labeled image')

    plt.show()

def saving_img(img, directory = 'labels.png'):
    """Function for saving labeled image into given directory.

    Args:
        img (numpy array): labeled image
        directory (str, optional): path to directory

    Returns:
        string: path to file
    """

    ind = np.argwhere(img == 0)
    color_map = plt.get_cmap('hsv', lut = (np.amax(img)+1))
    img = color_map(img)
    img = img*255
    img[ind[:, 0], ind[:, 1], 3] = 0
    cv2.imwrite('labels_transparent.png', img)
    img[ind[:, 0], ind[:, 1], :] = 0
    img[:, :, 3] = 255
    cv2.imwrite(directory, img)

    filename = directory

    return filename


def calculation(labeled, scale, type):
    """_summary_

    Args:
        labeled (numpy array): labeled image
        scale (int): microscope image scale
        type (string): nanoparticles or nanorods

    Returns:
        _type_: _description_
    """

    sizes = []
    sum_sizes = 0

    if type == 'Nanoparticles':

        '''
        regionprops_table(image, properties)
        '''
        for region in regionprops(labeled):

            diameter = (4 / math.pi *
                            region.area)**(1/2)
            sizes.append(diameter)
            sum_sizes = sum_sizes + diameter

    if type == 'Nanorods':
        
        for region in regionprops(labeled):

            sizes.append(region.area)
            sum_sizes = sum_sizes + region.area

    sizes_arr = np.array(sizes)

    fig = plt.hist(sizes_arr)
    plt.title('Histogram of sizes of NPs')
    plt.xlabel('size [px]')
    plt.ylabel('frequency')
    plt.savefig('histogram.png')

    avg = round(sum_sizes / len(sizes), 4)

    return sizes_arr, avg



if __name__ == '__main__':

    img_path = r'C:\Users\drakv\Desktop\fbmi\projekt\projekt\app\AuNR660_009.jpg'

    img_raw = loading_img(img_path)

    img_filtered = filtering_img(img_raw)

    binary = edges_operations(img_filtered)

    labels, distance = watershed_transform(binary)

    ploting_img(img_filtered, binary, distance, labels)