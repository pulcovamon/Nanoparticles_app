import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import json

from skimage import filters, morphology, img_as_ubyte
from skimage.io import imread
from skimage.filters import threshold_otsu, threshold_minimum
from skimage.color import rgb2gray, gray2rgb
from skimage.segmentation import watershed
from skimage.feature import peak_local_max, canny
from skimage.transform import (
    rescale, hough_circle, hough_circle_peaks, hough_ellipse
)
from skimage.measure import regionprops_table
from skimage.draw import ellipse_perimeter
import json
import os
import re
import argparse

# () /


def load_inputs(img_path):
    """Function for loading image names, scales from microscope and types of particles

    Args:
        img_path (path): path to directory with images
        json_path (path): path to json folder

    Returns:
        dict: dictionary with path to images, scales and types
    """
    for root, dirs, files in os.walk(img_path):
        for file in files:
            if file.endswith('.json'):
                json_path = img_path + '/' + file

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
    width = img_raw.shape[1]

    line = img_raw[-100:, int(width/2):, :]
    line = cv2.cvtColor(line, cv2.COLOR_RGB2GRAY)
    line = cv2.Canny(line, 0, 255)
    lines = cv2.HoughLinesP(line, 1, np.pi/180, threshold=50,
                            minLineLength=100, maxLineGap=10)
    lengths = [item[0][2]-item[0][0] for item in lines]
    pixel_size = scale / max(lengths)

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




def watershed_transform(img_filtered, np_type):
    """Function for labeling of image using calculated 
    distance map and markers for watershed transform

    Args:
        erode (numpy.ndarray): binary image
        region_fill (numpy.ndarray): binary image

    Returns:
        numpy.ndarray: labeled image
    """
    if np_type == 'nanoparticles':
        thresh = threshold_minimum(img_filtered)
    else:
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




def hough_segmentation(img_filtered, pixel_size):
    """Segmentation using hough circle and elipse transform

    Args:
        img_filtered (numpy array): grayscale image

    Returns:
        numpy array, list: RGB image with plotted circles/elipses,
                            list of NP parameters
    """

    canny_edge = canny(img_filtered, sigma = 2)

    start = max(10, 5/pixel_size)
    end = min(100, 50/pixel_size)
    hough_radii = np.arange(start, end, 2)
    hough_res = hough_circle(canny_edge, hough_radii)

    accums, x, y, r = hough_circle_peaks(hough_res, hough_radii,
                                min_xdistance=int(10),
                                min_ydistance=int(10))
    x = np.uint16(np.around(x)) 
    y = np.uint16(np.around(y))
    r = np.uint16(np.around(r))

    circles = [(x[i], y[i], r[i]) for i in range(len(x)) if accums[i] > 0.35]

    img = np.zeros((img_filtered.shape[0], img_filtered.shape[1], 3))
    img[:, :, 0] = img_filtered
    img[:, :, 1] = img_filtered
    img[:, :, 2] = img_filtered

    thresh = threshold_otsu(img_filtered)

    for x, y, r in circles:
        if img_filtered[x, y] > thresh:
            circles.remove((x, y, r))
        else:
            cv2.circle(img, (x, y), r, (0, 255, 0), 1)
            cv2.circle(img, (x, y), 1, (0, 255, 0), 2)

    return img, circles




def hough_nanorods(img_filtered, pixel_size):

    edges = canny(img_filtered, sigma=2)
    image_rgb = np.zeros((img_filtered.shape[0], img_filtered.shape[1], 3))
    image_rgb[:, :, 0] = img_filtered
    image_rgb[:, :, 1] = img_filtered
    image_rgb[:, :, 2] = img_filtered
    
    result = hough_ellipse(edges, accuracy=50, threshold=150,
                       min_size=400, max_size=500)
    result.sort(order='accumulator')

    # Estimated parameters for the ellipse
    best = list(result[-1])
    yc, xc, a, b = [int(round(x)) for x in best[1:5]]
    orientation = best[5]

    # Draw the ellipse on the original image
    cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
    image_rgb[cy, cx] = (0, 0, 255)
    # Draw the edge (white) and the resulting ellipse (red)
    edges = gray2rgb(img_as_ubyte(edges))
    edges[cy, cx] = (250, 0, 0)

    fig2, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(8, 4),
                                    sharex=True, sharey=True)

    ax1.set_title('Original picture')
    ax1.imshow(image_rgb)

    ax2.set_title('Edge (white) and result (red)')
    ax2.imshow(edges)

    plt.show()




def ploting_img(images, names, method):
    """Function for plotting images from segmentation proces

    Args:
        img (numpy.ndarray): image for plotting
        file_name (str): path to image
        method (str): segmentation method
        number (int): number of images
        current_num (int): number of this image
    """
    for i in range(len(images)):
        width = round(len(images)/2) + 1
        name = re.split('/|\.', names[i])

        plt.subplot(2, width, i+1)

        if method == 'watershed':
            plt.imshow(images[i], cmap='tab20')
        else:
            plt.imshow(images[i])

        plt.title(name[-2])
    plt.show()




def saving_img(img, file_name, directory = 'results'):
    """Function for saving result image into given directory.

    Args:
        img (numpy array): labeled image
        directory (str, optional): path to directory

    Returns:
        string: path to file
    """
    name = re.split('/', file_name)
    res_path = directory + '/' + name[-1]

    ind = np.argwhere(img == 0)
    color_map = plt.get_cmap('hsv', lut = (np.amax(img)+1))
    img = color_map(img)
    img = img*255
    img[ind[:, 0], ind[:, 1], :] = 0
    cv2.imwrite(res_path, img)

    return res_path




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

        sizes = 2*(props['area_convex']/np.pi)**2



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

        sizes = [[props['axis_major_length']], [props['axis_minor_length']]]


    avg = sum(props['area_convex']) / len(props['area_convex'])
    diameter = 2*(avg/np.pi)**(1/2)

    return diameter, sizes



def calcultaion_hough_transform(circles, pixel_size, np_type):
    """Calculation of NP mean area and radius.

    Args:
        circles (list): list of NP parameters
        pixel_size (float): size of one pixel in nm
        np_type (string): nanoparticles or nanorods

    Returns:
        float, float: mean diameter and mean area of NP
    """

    diameter = [r*2*pixel_size for x, y, r in circles]
    area = [r*pixel_size*np.pi**2 for x, y, r in circles]

    mean_diameter = np.mean(diameter)

    return mean_diameter, diameter



def histogram_sizes(sizes, file_name, np_type):
    """Function for creating histogram of sizes of NP

    Args:
        sizes (list): list of NP sizes
    """

    file_name = file_name[:-4] + 'hist' + file_name[-4:]

    if np_type == 'nanoparticles':
        plt.hist(sizes)
        plt.title('Histogram of sizes of NPs')
        plt.xlabel('diameter [nm]')
        plt.ylabel('frequency')
        plt.savefig(file_name)
        plt.clf()

    elif np_type == 'nanorods':
        plt.hist(sizes[0])
        plt.hist(sizes[1])
        plt.legend(['major axis', 'minor axis'])
        plt.title('Histogram of sizes of NRs')
        plt.xlabel('axis length [nm]')
        plt.ylabel('frequency')
        plt.savefig(file_name)
        plt.clf()




def read_args():
    """Function for command line arguments

    Returns:
        dictionary: path to image folder and method of segmentation
    """

    parser = argparse.ArgumentParser(description='segmentation of NPs',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-f', '--file', type=str, default='images',
                        help='path to images (default: images)')
    parser.add_argument('-m', '--method', type=str, default='watershed',
                        help='segmentation method (default: watershed)')
    args = parser.parse_args()
    config = vars(args)
    print(config)

    return config




if __name__ == '__main__':
    config = read_args()

    input_description = load_inputs(config['file'])
    method = config['method']

    if method != ('watershed' or 'hough'):
        raise Exception('unknown method')

    images = []
    names = []

    for image in input_description:
        scale = int(input_description[image][0])
        np_type = input_description[image][1]
        img_path = input_description[image][2]

        img_raw, pixel_size = loading_img(
            img_path, scale)

        img_filtered, pixel_size = filtering_img(
            img_raw, scale, np_type, pixel_size)

        if method == 'watershed':
            img, distance = watershed_transform(img_filtered, np_type)
            diameter, sizes = calculation_watershed(img, pixel_size,
                                                    np_type)

        elif method == 'hough':
            if np_type == 'nanoparticles':
                img, circles = hough_segmentation(img_filtered, pixel_size)
                diameter, sizes = calcultaion_hough_transform(circles,
                                                    pixel_size, np_type)
            elif np_type == 'nanorods':
                pass

        images.append(img)
        names.append(img_path)

        file_name = saving_img(img, img_path)
        histogram_sizes(sizes, file_name, np_type)

        print('saving into:', file_name)
    ploting_img(images, names, method)