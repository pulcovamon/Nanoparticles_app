import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import json

from skimage import filters, morphology
from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from skimage.segmentation import watershed
from skimage.feature import peak_local_max, canny
from skimage.transform import (
    rescale, hough_circle, hough_circle_peaks, hough_line
)
from skimage.measure import regionprops_table
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

    return img_filtered, pixel_size



def watershed_transform(img_filtered):
    """Function for labeling of image using calculated 
    distance map and markers for watershed transform

    Args:
        erode (numpy.ndarray): binary image
        region_fill (numpy.ndarray): binary image

    Returns:
        numpy.ndarray: labeled image
    """
    thresh = threshold_otsu(img_filtered)
    binary = img_filtered < thresh

    distance = ndi.distance_transform_edt(binary)

    coords = peak_local_max(
        distance, footprint = np.ones((3, 3)), min_distance = 20)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True

    markers, _ = ndi.label(mask)

    labels = watershed(-distance, markers, mask = binary)

    return labels, distance


def hough_segmentation(img_filtered, np_type):
    """Segmentation using hough circle and elipse transform

    Args:
        img_filtered (numpy array): grayscale image

    Returns:
        numpy array, list: RGB image with plotted circles/elipses,
                            list of NP parameters
    """

    canny_edge = canny(img_filtered, sigma = 2)

    hough_radii = np.arange(10, 100, 2)
    hough_res = hough_circle(canny_edge, hough_radii)

    accums, x, y, r = hough_circle_peaks(hough_res, hough_radii,
            min_xdistance=10, min_ydistance=10)
    x = np.uint16(np.around(x)) 
    y = np.uint16(np.around(y))
    r = np.uint16(np.around(r))

    circles = [(x[i], y[i], r[i]) for i in range(len(x)) if accums[i] > 0.3]

    img = np.zeros((img_filtered.shape[0], img_filtered.shape[1], 3))
    img[:, :, 0] = img_filtered
    img[:, :, 1] = img_filtered
    img[:, :, 2] = img_filtered

    for x, y, r in circles:
        cv2.circle(img, (x, y), r, (0, 255, 0), 1)
        cv2.circle(img, (x, y), 1, (0, 255, 0), 2)

    return img, circles


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


def calculation_watershed(labeled, pixel_size, np_type):
    """Calculation of mean radius and area of NP.

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


    avg = round(sum(props['area_convex']) / len(props['area_convex']), 4)

    return avg


def calcultaion_hough_transform(circles, pixel_size, np_type):
    """Calculation of NP mean area and radius.

    Args:
        circles (list): list of NP parameters
        pixel_size (float): size of one pixel in nm
        np_type (string): nanoparticles or nanorods

    Returns:
        float, float: mean radius and mean area of NP
    """

    radius = [r*pixel_size for x, y, r in circles]
    area = [r*pixel_size*np.pi**2 for x, y, r in circles]

    mean_radius = np.mean(radius)
    mean_area = np.mean(area)

    return mean_radius, mean_area


def histogram_sizes(sizes):
    """Function for creating histogram of sizes of NP

    Args:
        sizes (list): list of NP sizes
    """

    plt.hist(sizes)
    plt.title('Histogram of sizes of NPs')
    plt.xlabel('size [px]')
    plt.ylabel('frequency')
    plt.savefig('results/histogram.png')
    plt.clf()


if __name__ == '__main__':

    input_description = load_inputs(
        '/home/monika/Desktop/project/Nanoparticles_app/images',
        '/home/monika/Desktop/project/Nanoparticles_app/images/scales.json')

    for image in input_description:
        scale = int(input_description[image][0])
        np_type = input_description[image][1]
        img_path = input_description[image][2]

        img_raw, pixel_size = loading_img(
            img_path, scale)

        img_filtered, pixel_size = filtering_img(
            img_raw, scale, np_type, pixel_size)
        print(pixel_size)

        labels, distance = watershed_transform(img_filtered)

        file_name = saving_img(labels)

        avg = calculation_watershed(labels, pixel_size,
                                                    np_type)

        if np_type == 'nanoparticles':
            img, circles = hough_segmentation(img_filtered, np_type)
            radius, area = calcultaion_hough_transform(circles,
                                                pixel_size, np_type)
