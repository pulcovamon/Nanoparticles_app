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
    img_inverted = util.invert(img_raw)

    img_filtered = filters.median(
                    img_raw, morphology.disk(5))

    sobel_edges = filters.sobel(img_filtered)

    return img_filtered, sobel_edges



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

    coords = peak_local_max(distance, footprint = np.ones((3, 3)), min_distance = 20)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True

    markers, _ = ndi.label(mask)

    labels = watershed(-distance, markers, mask = region_fill)

    return labels


def ploting_img(img_median, sobel_edges, region_fill, labels):
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
    plt.imshow(sobel_edges, cmap = 'gray')
    plt.title('Edges detected using Sobel operator')

    plt.subplot(2, 2, 3)
    plt.imshow(region_fill, cmap = 'gray')
    plt.title('Binary image')

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
        params = cv2.SimpleBlobDetector_Params()

        params.filterByCircularity = True
        params.minCircularity = 0.1
        params.filterByConvexity = True
        params.minConvexity = 0.3
        params.filterByInertia = True
        params.minInertiaRatio = 0.5

        img = cv2.imread('labels.png', 0)
        img = 255 - img

        detector = cv2.SimpleBlobDetector_create(params)
        keypoint = detector.detect(img)
        blank = np.zeros((1, 1))
        blobs = cv2.drawKeypoints(img, keypoint, blank, (0, 0, 255),
                                    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imwrite('blobs.png', blobs)
        '''

        '''
        regionprops_table(image, properties)
        '''
        for region in regionprops(labeled):

            diameter = (4 / math.pi * region.area)**(1/2)
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

    img_path = r'C:\Users\drakv\Desktop\fbmi\projekt\projekt\app\AuNP20nm_004.jpg'

    img_raw = loading_img(img_path)

    img_median, sobel_edges = filtering_img(img_raw)

    region_fill, erode = edges_operations(sobel_edges)

    labels = watershed_transform(erode, region_fill)

    ploting_img(img_median, sobel_edges, region_fill, labels)