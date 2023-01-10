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
from skimage.transform import rescale, hough_circle, hough_circle_peaks, hough_ellipse
from skimage.measure import regionprops_table, label
from skimage.draw import ellipse_perimeter
from skimage.morphology import erosion, remove_small_holes, remove_small_objects
import json
import os
import re
import argparse
from copy import deepcopy, copy
from time import time

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
            if file.endswith(".json"):
                json_path = img_path + "/" + file

    with open(json_path) as json_file:
        input_description = json.load(json_file)

        for key in input_description:
            input_description[key].append(os.path.join(img_path, key))

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

    line = img_raw[-100:, int(width / 2) :, :]
    line = cv2.cvtColor(line, cv2.COLOR_RGB2GRAY)

    mask = line < 10
    # try funciton remove small objects
    mask = remove_small_objects(mask)
    indices = np.where(mask)
    length = max(indices[1]) - min(indices[1])
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

    img_filtered = filters.median(img_raw, kernel)

    return img_filtered, pixel_size


def binarizing(img, np_type):
    """Function for finding threshold and binarize image

    Args:
        img (numpy.ndarray): single-channel image
        np_type (string): nanoparticles or nanorods

    Returns:
        numpy array: binary image
    """

    if np_type == "nanoparticles":
        thresh = threshold_minimum(img)
    else:
        thresh = threshold_otsu(img)

    binary = img < thresh
    binary = remove_small_holes(binary)
    
    return binary


def watershed_transform(binary):
    """Function for labeling of image using calculated
    distance map and markers for watershed transform

    Args:
        binary (numpy.ndarray): binary image

    Returns:
        numpy.ndarray: labeled image
    """
    distance = ndi.distance_transform_edt(binary)

    coords = peak_local_max(distance, footprint=np.ones((3, 3)), min_distance=20)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True

    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=binary)

    return labels


def segmentation(img, binary, np_type, pixel_size):

    labels = watershed_transform(binary)
    sizes, props_watershed = calculation_watershed(labels, pixel_size, np_type)
    labels, props_watershed = filter_blobs(labels, sizes, props_watershed)
    props_ht = find_overlaps(img, binary, sizes, np_type, pixel_size)
    img, sizes = find_duplicity(labels, props_ht, props_watershed, pixel_size)

    return img, sizes


def overlay_images(raw, labels):

    raw = raw/255
    r = rescale(labels[:, :, 0], 2)
    g = rescale(labels[:, :, 1], 2)
    b = rescale(labels[:, :, 2], 2)
    rgb = np.zeros((r.shape[0], r.shape[1], 3))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    img = (7*raw + 3*rgb)/10

    return img



def find_duplicity(labels, props_ht, props_watershed, pixel_size):
    """Function for

    Args:
        labels (_type_): _description_
        props_ht (_type_): _description_
        props_watershed (_type_): _description_

    Returns:
        _type_: _description_
    """

    size_ht = []
    for cx, cy, r in props_ht:
        val = labels[cy, cx]
        props_watershed = [i for i in props_watershed if i[0] != val]
        d = 2*r
        size_ht.append(d)

    diameter = lambda a: 2 * (a / np.pi)**(1/2)
    size_watershed = [diameter(i[1]) for i in props_watershed]
    sizes = size_ht + size_watershed
    sizes = [round(pixel_size * i, 3) for i in sizes]

    max_val = np.max(labels)
    multiplicator = int(255 / max_val)
    labels = np.uint8(labels * multiplicator)
    color_map = plt.get_cmap("nipy_spectral")
    img = color_map(labels)
    img, _ = draw_circles(img, props_ht)

    return img, sizes


def find_overlaps(img, binary, sizes, np_type, pixel_size):

    labels = label(binary, background=0)
    _, props = calculation_watershed(labels, pixel_size, np_type)

    median = calc_median(deepcopy(sizes))
    props_ht = []

    for number, size in props:
        if size > 1.5 * median:
            indices = np.argwhere(labels == number)
            x_min = max(np.min(indices[:, 0]) - 10, 0)
            y_min = max(np.min(indices[:, 1]) - 10, 0)
            x_max = min(np.max(indices[:, 0]) + 10, labels.shape[0]-1)
            y_max = min(np.max(indices[:, 1]) + 10, labels.shape[1]-1)
            roi = img[x_min:x_max, y_min:y_max]
            circles = hough_segmentation(roi, pixel_size, np_type)
            _, area = calcultaion_hough_transform(circles, pixel_size, np_type)
            circles = filter_circles(roi, circles, area)
            for circle in circles:
                cx = circle[0] + y_min
                cy = circle[1] + x_min
                r = circle[2]
                props_ht.append((cx, cy, r))

    return props_ht


def filter_blobs(labels, sizes, props):
    """Function for eliminate oversegmented blobs

    Args:
        labels (numpy.ndarray): labeled image from watershed
        sizes (list): list of sizes from watershed transform
        props (list): list of tuples with number of label
                        and particular size from watershed

    Returns:
        numpy.ndarray, list: labeled image, list of properties
    """

    median = calc_median(deepcopy(sizes))

    for number, size in props:
        if size < median / 2:
            props.remove((number, size))
            labels[labels == number] = 0

    return labels, props


def hough_segmentation(img, pixel_size, np_type):
    """Segmentation using hough circle and elipse transform

    Args:
        img_filtered (numpy.ndarray): grayscale image

    Returns:
        numpy.ndarray, list: RGB image with plotted circles/elipses,
                            list of NP parameters
    """

    canny_edge = canny(img, sigma=2)

    start = max(10, 5 / pixel_size)
    end = min(100, 50 / pixel_size)
    hough_radii = np.arange(start, end, 2)
    hough_res = hough_circle(canny_edge, hough_radii)

    accums, x, y, r = hough_circle_peaks(
        hough_res, hough_radii, min_xdistance=int(10), min_ydistance=int(10)
    )
    x = np.uint16(np.around(x))
    y = np.uint16(np.around(y))
    r = np.uint16(np.around(r))

    circles = [(x[i], y[i], r[i]) for i in range(len(x)) if accums[i] > 0.25]

    return circles


def draw_circles(img, circles):
    """Function for plotting circles into original image

    Args:
        gray (numpy.ndarray): RGB image
        circles (list): list with tuples with center
                        indices and radius of circles
        area (list): list of calculated areas in nm

    Returns:
        numpy array, list: image with circles,
                        list with tuples with center
                        indices and radius of circles
    """

    for x, y, r in circles:
        cv2.circle(img, (x, y), r, (255, 0, 255), 1)
        #cv2.circle(img, (x, y), 1, (0, 0, 0), 2)

    return img, circles


def filter_circles(gray, circles, area):
    """Function for deleting too small and too bright
    and circles not fitting into image

    Args:
        gray (numpy.ndarray): single-channel image
        circles (list): list with tuples with center
                        indices and radius of circles
        area (list): list of calculated areas in nm

    Returns:
        list: list with tuples with center
                        indices and radius of circles
    """

    dims = gray.shape
    thresh = threshold_minimum(gray)

    i = 0
    j = 0
    first_cycle = True
    while j < len(circles):
        if i == len(circles):
            i = 0
            first_cycle = False
        cx = int(circles[i][0])
        cy = int(circles[i][1])
        r = int(circles[i][2])
        median_area = calc_median(deepcopy(area))
        mean_area = np.mean(area)
        if first_cycle:
            pixels = pixels_in_circle(gray, cx, cy, r)
            median_value = calc_median(copy(pixels))
            if median_value > thresh:
                circles.pop(i)
                area.pop(i)
            elif cx - r < 0 or cy - r < 0 or cx + r > dims[1] or cy + r > dims[0]:
                circles.pop(i)
                area.pop(i)
            else:
                i += 1
        else:
            if area[i] <= median_area / 1.5 or area[i] <= mean_area / 1.5:
                circles.pop(i)
                area.pop(i)
                j = 0
            else:
                i += 1
                j += 1

    return circles


def pixels_in_circle(img, cx, cy, r):
    """Function for finding all pixel values inside circle

    Args:
        img (numpy.ndarray): single-channel image
        cx (int): first index of center
        cy (int): second index of center
        r (int): radius

    Returns:
        list: pixel values inside circle
    """

    pixels = []

    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            dx = x - cx
            dy = y - cy
            distance = dx**2 + dy**2

            if distance <= r**2:
                pixels.append(img[y, x])

    return pixels


def calc_median(data_list):
    """Function for calculating median

    Args:
        data_list (list): list with numbers

    Returns:
        float: median value
    """

    n = len(data_list)
    data_list.sort()
    index = int(n / 2)
    if n % 2 == 0:
        median = (data_list[index] + data_list[index - 1]) / 2
    else:
        median = data_list[index]

    return median



def ploting_img(images, names):
    """Function for plotting images from segmentation proces

    Args:
        img (numpy.ndarray): image for plotting
        names(str): path to images
    """
    for i in range(len(images)):
        width = round(len(images) / 2) + 1
        name = re.split("/|\.", names[i])

        plt.subplot(2, width, i + 1)
        plt.imshow(images[i])
        plt.title(name[-2])

    plt.show()


def saving_img(img, file_name, directory="results"):
    """Function for saving result image into given directory.

    Args:
        img (numpy array): labeled image
        directory (str, optional): path to directory

    Returns:
        string: path to file
    """
    name = re.split("/", file_name)
    res_path = directory + "/" + name[-1]

    ind = np.argwhere(img == 0)
    color_map = plt.get_cmap("hsv", lut=(np.amax(img) + 1))
    img = color_map(img)
    img = img * 255
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

    if np_type.lower() == "nanoparticles":
        props = regionprops_table(
            labeled, properties=["label", "area_convex", "equivalent_diameter_area"]
        )

    if np_type.lower() == "nanorods":
        props = regionprops_table(
            labeled,
            properties=[
                "label",
                "area_convex",
                "axis_major_length",
                "axis_minor_length",
            ],
        )

    avg = sum(props["area_convex"]) / len(props["area_convex"])
    diameter = 2 * (avg / np.pi) ** (1 / 2)

    selected = [
        (props["label"][i], props["area_convex"][i]) for i in range(len(props["label"]))
    ]
    sizes = props["area_convex"]

    return sizes, selected


def calcultaion_hough_transform(circles, pixel_size, np_type):
    """Calculation of NP mean area and radius.

    Args:
        circles (list): list of NP parameters
        pixel_size (float): size of one pixel in nm
        np_type (string): nanoparticles or nanorods

    Returns:
        float, float: mean diameter and mean area of NP
    """

    diameter = [r * 2 * pixel_size for x, y, r in circles]
    area = [(r * pixel_size) ** 2 * np.pi for x, y, r in circles]

    return diameter, area


def histogram_sizes(sizes, file_name, np_type):
    """Function for creating histogram of sizes of NP

    Args:
        sizes (list): list of NP sizes
    """

    file_name = file_name[:-4] + "hist" + file_name[-4:]

    if np_type == "nanoparticles":
        plt.hist(sizes)
        plt.title("Histogram of sizes of NPs")
        plt.xlabel("diameter [nm]")
        plt.ylabel("frequency")
        plt.savefig(file_name)
        plt.clf()

    elif np_type == "nanorods":
        plt.hist(sizes[0])
        plt.hist(sizes[1])
        plt.legend(["major axis", "minor axis"])
        plt.title("Histogram of sizes of NRs")
        plt.xlabel("axis length [nm]")
        plt.ylabel("frequency")
        plt.savefig(file_name)
        plt.clf()


def read_args():
    """Function for command line arguments

    Returns:
        dictionary: path to image folder and method of segmentation
    """

    parser = argparse.ArgumentParser(
        description="segmentation of NPs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default="images",
        help="path to images (default: images)",
    )
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        default="watershed",
        help="segmentation method (default: watershed)",
    )
    parser.add_argument(
        "-g",
        "--graph",
        type=bool,
        default=False,
        help="Plotting image (default: False)",
    )
    args = parser.parse_args()
    config = vars(args)
    print(config)

    return config


if __name__ == "__main__":
    config = read_args()

    input_description = load_inputs(config["path"])
    method = config["method"]
    graph = config["graph"]

    if method != "watershed":
        raise Exception("unknown method")

    images = []
    names = []

    for image in input_description:
        scale = int(input_description[image][0])
        np_type = input_description[image][1]
        img_path = input_description[image][2]

        img_raw, pixel_size = loading_img(img_path, scale)
        img_filtered, pixel_size = filtering_img(img_raw, scale, np_type, pixel_size)
        binary = binarizing(img_filtered, pixel_size)

        if np_type == 'nanoparticles':
            labels, sizes = segmentation(img_filtered, binary, np_type, pixel_size)
        elif np_type == 'nanorods':
            pass
        else:
            raise Exception("Invalid NP type")

        img = overlay_images(img_raw, labels)

        images.append(img)
        names.append(img_path)

        file_name = saving_img(img, img_path)
        histogram_sizes(sizes, file_name, np_type)

        print("saving into:", file_name)
        
    if graph == True:
        ploting_img(images, names)
