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
from skimage.transform import (
    rescale, hough_circle, hough_circle_peaks, hough_line
)
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


def loading_img(img_path, scale):
    """Function for loading image from given file and croping it

    Args:
        img_path (str): path to image file
        scale (int): scale from microscope

    Returns:
        numpy.ndarray: RGB image
        flaot: size of pixel in raw image
    """

    img_raw = imread(img_path)

    line = img_raw[-100:, :]
    line = rgb2gray(line)
    thresh = threshold_otsu(line)
    line = line > thresh
    tested_angles = np.linspace(-np.pi / 2,
                np.pi / 2, 360, endpoint=False)
    _, _, distance = hough_line(line, theta=tested_angles)
    length = distance[-1] - distance[0]
    pixel_size = scale / length

    img_raw = img_raw[0:-100, :]

    return img_raw, pixel_size



def filtering_img(img_raw, scale, type, pixel_size):
    """Function for bluring image and edge detection

    Args:
        img_raw (numpy.ndarray): RGB image
        scale (int): scale from microscopy image
        type (str): nanoparticles or nanorods
        pixel_size (float): size of one pixel in raw image in nm

    Returns:
        numpy.ndarray: grayscale image
        float: rescaled pixel size

    """
    img_raw = rgb2gray(img_raw)
    img_raw = rescale(img_raw, 0.5)
    pixel_size *= 2

    if scale < 200:
        kernel = morphology.disk(5)
    else:
        kernel = morphology.disk(3)
    
    img_filtered = filters.median(
                    img_raw, kernel)

    thresh = threshold_otsu(img_filtered)
    binary = img_filtered < thresh

    return binary, pixel_size



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
    cv2.imwrite('results/labels_transparent.png', img)
    img[ind[:, 0], ind[:, 1], :] = 0
    img[:, :, 3] = 255
    cv2.imwrite(directory, img)

    filename = directory

    return filename


def calculation(labeled, pixel_size, np_type):
    """_summary_

    Args:
        labeled (numpy array): labeled image
        pixel_size (float): size of one pixel in image
        np_type (string): nanoparticles or nanorods

    Returns:
        _type_: _description_
    """

    if np_type.lower() == 'nanoparticles':

        props = regionprops_table(labeled, properties =
            ['label', 'area_convex', 'equivalent_diameter_area'])

        with open('results/props.txt', 'w') as txt_file:
            txt_file.write('number area diameter')

            for i in range(len(props['label'])):
                txt_file.write('\n')
                for key_val in props.keys():
                    if key_val != 'label':
                        props[key_val][i] *= pixel_size
                    txt_file.write(str(props[key_val][i])+' ')



    if np_type.lower() == 'nanorods':

        props = regionprops_table(labeled, properties =
                ['label', 'area_convex', 'axis_major_length',
                                        'axis_minor_length'])

        with open('results/props.txt', 'w') as txt_file:
            txt_file.write('number area major_axis minor_axis')

            for i in range(len(props['label'])):
                txt_file.write('\n')
                for key_val in props.keys():
                    if key_val != 'label':
                        props[key_val][i] *= pixel_size
                    txt_file.write(str(props[key_val][i])+' ')



    plt.hist(props['area_convex'])
    plt.title('Histogram of sizes of NPs')
    plt.xlabel('size [px]')
    plt.ylabel('frequency')
    plt.savefig('results/histogram.png')
    plt.clf()

    avg = round(sum(props['area_convex']) / len(props['area_convex']), 4)

    return avg



if __name__ == '__main__':

    input_description = load_inputs(
        '/home/monika/Desktop/project/Nanoparticles_app/images',
        '/home/monika/Desktop/project/Nanoparticles_app/images/scales.json')

    for image in input_description:

        img_path = input_description[image][2]

        img_raw, pixel_size = loading_img(
            img_path, int(input_description[image][0]))

        binary, pixel_size = filtering_img(
            img_raw, int(input_description[image][0]),
            input_description[image][1], pixel_size)

        labels, distance = watershed_transform(binary)

        file_name = saving_img(labels)

        avg = calculation(labels, pixel_size,
            input_description[image][1])