import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import json
import math

from skimage import filters, morphology, util
from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.transform import rescale, hough_circle, hough_circle_peaks
from skimage.measure import regionprops, regionprops_table
import json
import os

# () /

def load_inputs(img_path, json_path):
    """Function for loading image names, scales from microscope and types of particles

    Args:
        img_path (path): path to directory with images
        json_path (path): path to json folder

    Returns:
        dict: dictionary with path to images, scales and types
    """

    with open(json_path) as json_file:
        input_description = json.load(json_file)

        for key in input_description:
            input_description[key].append(
                os.path.join(img_path, key))

    return input_description


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
                    img_raw, morphology.disk(3))

    thresh = threshold_otsu(img_filtered)
    binary = img_filtered < thresh

    return binary



def watershed_transform(binary):
    """Function for labeling of image using calculated 
    distance map and markers for watershed transform

    Args:
        erode (numpy.ndarray): binary image
        region_fill (numpy.ndarray): binary image

    Returns:
        numpy.ndarray: labeled image
    """

    distance = ndi.distance_transform_edt(binary)

    coords = peak_local_max(
        distance, footprint = np.ones((3, 3)), min_distance = 20)
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

def saving_img(img, directory = '/results/labels.png'):
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
    cv2.imwrite('/results/labels_transparent.png', img)
    img[ind[:, 0], ind[:, 1], :] = 0
    img[:, :, 3] = 255
    cv2.imwrite(directory, img)

    filename = directory

    return filename


def calculation(labeled, scale, np_type):
    """_summary_

    Args:
        labeled (numpy array): labeled image
        scale (int): microscope image scale
        np_type (string): nanoparticles or nanorods

    Returns:
        _type_: _description_
    """

    sizes = []
    sum_sizes = 0

    if np_type == 'Nanoparticles':

        props = regionprops_table(labeled, properties =
            ['label', 'area_convex', 'equivalent_diameter_area'])

        with open('/results/props.txt', 'w') as txt_file:
            txt_file.write('number area diameter')

            for i in range(len(props['label'])):
                txt_file.write('\n')
                for key_val in props.keys():
                    txt_file.write(str(props[key_val][i])+' ')



    if np_type == 'Nanorods':

        props = regionprops_table(labeled, properties =
                ['label', 'area_convex', 'axis_major_length',
                                        'axis_minor_length'])

        with open('/results/props.txt', 'w') as txt_file:
            txt_file.write('number area major_axis minor_axis')

            for i in range(len(props['label'])):
                txt_file.write('\n')
                for key_val in props.keys():
                    txt_file.write(str(props[key_val][i])+' ')



    plt.hist(props['area_convex'])
    plt.title('Histogram of sizes of NPs')
    plt.xlabel('size [px]')
    plt.ylabel('frequency')
    plt.savefig('/results/histogram.png')
    plt.clf()

    avg = round(sum(props['area_convex']) / len(props['area_convex']), 4)

    return avg



if __name__ == '__main__':

    input_description = load_inputs(
        '/home/monika/Desktop/project/Nanoparticles_app/images',
        '/home/monika/Desktop/project/Nanoparticles_app/images/scales.json')

    for image in input_description:

        img_path = input_description[image][2]

        img_raw = loading_img(img_path)

        binary = filtering_img(img_raw)

        labels, distance = watershed_transform(binary)

        ploting_img(img_raw, binary, distance, labels)