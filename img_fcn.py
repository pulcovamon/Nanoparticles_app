"""Analyze TEM images on gold nanoparticles"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import json
import os
import re
import argparse
from copy import deepcopy, copy
from skimage.io import imread
from skimage.filters import threshold_otsu, threshold_minimum, median
from skimage.color import rgb2gray
from skimage.segmentation import watershed, clear_border
from skimage.feature import peak_local_max, canny
from skimage.transform import rescale, hough_circle, hough_circle_peaks
from skimage.measure import regionprops_table, label
from skimage.morphology import (
    remove_small_holes, remove_small_objects, disk, erosion, dilation, opening, convex_hull_image, closing
)
from sklearn.cluster import KMeans
import random


def load_inputs(img_path):
    """Load image names, scales
        from microscope and types of particles

    Args:
        img_path (path): path to directory with images
        json_path (path): path to json folder

    Returns:
        dict: dictionary with path to images, scales and types
    """
    for _, dirs, files in os.walk(img_path):
        json_found = False
        for file in files:
            if file.endswith(".json"):
                if json_found:
                    raise Exception("multiple input descriptions")
                json_path = os.path.join(img_path, file)
                json_found = True

    if not json_found:
        raise Exception('json file not found')

    with open(json_path, mode='r') as json_file:
        input_description = json.load(json_file)

    for key in input_description.keys():
        if not input_description[key][0].isdigit():
            raise Exception("wrong input: size not number")

        input_description[key][1].lower()
        if (
            input_description[key][1] != "nanoparticles"
            and input_description[key][1] != "nanorods"
        ):
            raise Exception("wrong input: unknown type")

        input_description[key].append(os.path.join(img_path, key))

    return input_description


def loading_img(img_path, scale):
    """Load image from given file and croping it

    Args:
        img_path (str): path to image file
        scale (int): scale from microscope

    Returns:
        numpy.ndarray: RGB image
        flaot: size of pixel in raw image
    """
    try:
        img_raw = imread(img_path)
    except Exception as err:
        print(err)
        raise Exception("wrong input: image cannot be open")

    width = img_raw.shape[1]

    line = img_raw[-100:, int(width / 2) :, :]
    line = cv2.cvtColor(line, cv2.COLOR_RGB2GRAY)

    mask = line < 10
    mask = remove_small_objects(mask)
    indices = np.where(mask)
    length = max(indices[1]) - min(indices[1])
    global pixel_size
    pixel_size = scale / length

    img_raw = img_raw[0:-100, :]

    return img_raw, pixel_size


def filtering_img(img_raw, scale, np_type, pixel_size, is_bg):
    """Blur image and edge detection

    Args:
        img_raw (numpy.ndarray): RGB image
        scale (int): scale from microscopy image
        type (str): nanoparticles or nanorods
        pixel_size (float): size of one pixel in raw image in nm

    Returns:
        numpy.ndarray: grayscale image
        float: rescaled pixel size

    """
    img = rgb2gray(img_raw)

    if np_type == "nanorods":
        if scale < 200:
            kernel = disk(5)
        else:
            kernel = disk(3)
        img = median(img, kernel)
        if is_bg:
            img = background(img)

    img = rescale(img, 0.5)
    pixel_size *= 2

    if np_type == "nanoparticles":
        if scale < 200:
            kernel = disk(7)
        elif scale < 500:
            kernel = disk(5)
        else:
            kernel = disk(3)

        img = median(img, kernel)

    return img, pixel_size


def background(img):
    
    """Remove background from image

    Args:
        img (numpy.ndarray): single-channel image

    Returns:
        numpy.ndarray: image without background
    """

    img *= 255
    img = cv2.edgePreservingFilter(img.astype(np.uint8),
                                   flags=cv2.RECURS_FILTER,
                                   sigma_s=60,
                                   sigma_r=0.1)
    img = cv2.medianBlur(img.astype(np.uint8), 9)
    
    return img


def binarizing(img, np_type):
    """Find threshold, binarize image and perfrom morphological operations
        for clearing image, cannny edge detection for better separation.

    Args:
        img (numpy.ndarray): single-channel image
        np_type (string): nanoparticles or nanorods

    Returns:
        numpy array: binary image
    """
    thresh = threshold_otsu(img)

    binary = img < thresh
    binary = remove_small_holes(binary)
    binary = remove_small_objects(binary)

    img *= 255
    edges = cv2.Canny(img.astype(np.uint8), 20, 200)

    '''plt.subplot(2, 2, 1)
    plt.imshow(binary, cmap='gray')
    plt.subplot(2, 2, 2)
    plt.imshow(edges, cmap='gray')'''

    binary[edges == 255] = 0
    binary = cv2.morphologyEx(binary.astype(np.uint8),
                                    cv2.MORPH_ERODE, disk(5))
    binary = cv2.morphologyEx(binary.astype(np.uint8),
                                    cv2.MORPH_DILATE, disk(5))
    if np_type == 'nanorods':
        binary = ndi.binary_fill_holes(binary)
    binary = clear_border(binary)

    '''plt.subplot(2, 2, 3)
    plt.imshow(binary, cmap='gray')
    plt.show()'''
        
    return binary


def watershed_transform(binary, np_type, pixel_size):
    """Label image using calculated
    distance map and markers for watershed transform

    Args:
        binary (numpy.ndarray): binary image

    Returns:
        numpy.ndarray: labeled image
    """
    if np_type == 'nanoparticles':
        seeds = np.zeros(binary.shape, dtype=np.uint8)
        mask = binary.copy()

        while np.sum(mask) > 0:
            regions = erosion(mask)
            regions = label(regions)

            for i in range(1, np.max(regions) + 1):
                roi = regions == i
                convex_hull = convex_hull_image(roi)
                diff = convex_hull != roi
                diff = np.sum(diff.astype(np.float32))
                area = roi.astype(np.float32).sum()
                convex = (diff == 0 or 10 * diff < area) and area > 5 / pixel_size
                if convex:
                    seeds[roi] = 1
                    regions[regions == i] = 0
            mask = regions > 0

        seeds = dilation(seeds)

        distance = ndi.distance_transform_edt(binary)
        img = -distance

    else:
        distance = ndi.distance_transform_edt(binary)
        img = -distance
        mask =  binary
        for _ in range(10):
            mask = remove_small_holes(mask)
            mask = remove_small_objects(mask)
        for _ in range(10):
            mask = erosion(mask)
            seeds = mask

    markers, _ = ndi.label(seeds)
    labels = watershed(img, markers, mask=binary)

    return labels


def segmentation(img, binary, np_type, pixel_size):
    """Perform segmentation and calculate sizes

    Args:
        raw (numpy.ndarray): raw image (three channels)
        img (numpy.ndarray): gray image (one channel)
        binary (numpy.ndarray): binary mask
        np_type (str): nanoparticles or nanorods
        pixel_size (float): size of one pixel in nm

    Returns:
        numpy.ndarray, list: labeled image, ist of sizes
    """
    labels = watershed_transform(binary, np_type, pixel_size)
    sizes, props_watershed = calculation_watershed(labels, np_type)
    sizes = [i for i in sizes]

    if np_type == "nanoparticles":
        props_ht = find_overlaps(img, labels, sizes, np_type, pixel_size)
        sizes = find_duplicity(labels, props_ht, props_watershed, pixel_size)
    else:
        labels, props_watershed = filter_blobs(labels, sizes, props_watershed)
        sizes[0] = [i * pixel_size for i in sizes[1]]
        sizes[1] = [i * pixel_size for i in sizes[2]]
        props_ht = []

    return labels, sizes, props_ht


def result_image(raw, labels, np_type, props_ht):
    """Create result image

    Args:
        raw (numpy.ndarray): 3-channels raw image
        labels (numpy.ndarray): labeled image
        np_type (str)): nanoparticles or nanorods
        props_ht (list): list of tuples with coordinates
                        of centers and radii

    Returns:
        numpy.ndarray: 3-channels result image
    """
    max_val = np.max(labels)
    mix = [i for i in range(1, max_val + 1)]
    random.shuffle(mix)
    labels_new = labels.copy()

    for i in range(1, max_val + 1):
        labels_new[labels == i] = mix[i - 1]

    labels = np.ma.masked_where(labels_new == 0, labels_new)
    c_map = plt.get_cmap('gist_rainbow', max_val).copy()
    c_map.set_bad(color='black', alpha=None)
    labels = c_map(labels)
    img = overlay_images(raw, labels)

    if np_type == "nanoparticles":
        for x, y, r in props_ht:
            x *= 2
            y *= 2
            r *= 2
            cv2.circle(img, (x, y), r, (0, 0, 0), 2)

    return img


def overlay_images(raw, labels):
    """Overlay labeled image over raw image

    Args:
        raw (numpy.ndarray): raw image (three channels)
        labels (numpy.ndarray): segmented image (three channels)

    Returns:
        numpy.ndarray: combination of raw image and segment image
    """
    raw = raw / 255
    r = rescale(labels[:, :, 0], 2)
    g = rescale(labels[:, :, 1], 2)
    b = rescale(labels[:, :, 2], 2)
    rgb = np.zeros((r.shape[0], r.shape[1], 3))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    img = (raw + rgb) / 2
    img = np.uint8(img * 255)

    return img


def find_duplicity(labels, props_ht, props_wsh, pixel_size):
    """Eliminate duplicities in properties

    Args:
        labels (numpy.ndarray):
        props_ht (_type_): _description_
        props_watershed (_type_): _description_

    Returns:
        _type_: _description_
    """
    size_ht = []
    for cx, cy, r in props_ht:
        val = labels[cy, cx]
        props_wsh = [i for i in props_wsh if i[0] != val]
        d = 2 * r
        size_ht.append(d)

    diameter = lambda a: 2 * (a / np.pi) ** (1 / 2)
    size_wsh = [diameter(i[1]) for i in props_wsh]
    sizes = size_ht + size_wsh
    sizes = [round(pixel_size * i, 3) for i in sizes]

    return sizes


def detect_seeds(sizes):
    sizes = np.array(sizes).reshape(-1, 1)
    kmeans = KMeans(n_clusters=2).fit(sizes)
    centers = kmeans.cluster_centers_
    if centers[0] > 4*centers[1] or centers[1] > 4*centers[0]:
        seeds = True
    else:
        seeds = False

    return seeds, centers
        


def find_overlaps(img, labels, sizes,  np_type, pixel_size):
    """Find overlapping particles and perform
        circle hough transform on found area

    Args:
        img (numpy.ndarray): gray image
        binary (numpy.ndarray): binary mask
        sizes (list): sizes from watershed transform
        np_type (str): nanoparticles or nanorods
        pixel_size (float): size of one pixel in nm

    Returns:
        list: list pf tuples with coordinates of center
                and radii of hough circles
    """
    props_ht = []
    median = calc_median(sizes)

    for i in range(1, np.max(labels) + 1):
        roi = labels == i
        convex_hull = convex_hull_image(roi)
        diff = convex_hull != roi
        diff = np.sum(diff.astype(np.float32))
        area = roi.astype(np.float32).sum()
        if diff > area / 20:
            indices = np.argwhere(labels == i)
            x_min = max(np.min(indices[:, 0]) - 10, 0)
            y_min = max(np.min(indices[:, 1]) - 10, 0)
            w = labels.shape[0]
            h = labels.shape[1]
            x_max = min(np.max(indices[:, 0]) + 10, w - 1)
            y_max = min(np.max(indices[:, 1]) + 10, h - 1)
            roi = labels[x_min:x_max, y_min:y_max]
            roi = roi == i

            circles = hough_segmentation(roi, pixel_size, np_type)
            if len(circles) > 1:
                area = [(r * pixel_size) ** 2 * np.pi for x, y, r in circles]
                circles = filter_circles(roi, circles, area, median)
            if len(circles) > 1:
                circles = circles_inside_circles(circles)

            for circle in circles:
                cx = circle[0] + y_min
                cy = circle[1] + x_min
                r = circle[2]
                props_ht.append((cx, cy, r))
                props_ht = small_circles(props_ht)

    return props_ht


def filter_blobs(labels, sizes, props):
    """Eliminate oversegmented blobs

    Args:
        labels (numpy.ndarray): labeled image from watershed
        sizes (list): list of sizes from watershed transform
        props (list): list of tuples with number of label
                        and particular size from watershed

    Returns:
        numpy.ndarray, list: labeled image, list of properties"""
    

    if type(sizes) is list:
        median = calc_median(deepcopy(sizes[0]))
        mean = sum(sizes[0]) / len(sizes[0])
    else:
        median = calc_median(deepcopy(sizes))
        mean = sum(sizes) / len(sizes)

    avg = max(mean, median)

    i = 0
    while i < len(props):
        smaller = props[i][1] < avg / 1.5
        if smaller:
            labels[labels == props[i][0]] = 0
            props.remove((props[i][0], props[i][1]))
        else:
            i += 1   


    return labels, props


def hough_segmentation(img, pixel_size, np_type):
    """Segmentation using hough circle and elipse transform

    Args:
        img_filtered (numpy.ndarray): grayscale image

    Returns:
        numpy.ndarray, list: RGB image with plotted
                circles/elipses, list of NP parameters
    """
    canny_edge = canny(img, sigma=4)

    start = 10 / pixel_size
    end = 100 / pixel_size
    min_dist = 10/pixel_size
    hough_radii = np.arange(start, end, 1)
    hough_res = hough_circle(canny_edge, hough_radii)

    accums, x, y, r = hough_circle_peaks(
        hough_res, hough_radii, min_xdistance=int(min_dist), min_ydistance=int(min_dist)
    )
    x = np.uint16(np.around(x))
    y = np.uint16(np.around(y))
    r = np.uint16(np.around(r))

    a = accums
    l = len(x)
    circles = [(x[i], y[i], r[i]) for i in range(l) if a[i] > 0.2]

    return circles


def small_circles(props):
    """Delete small circles

    Args:
        props (list): list of tuples with center
                        indices and radius of circles

    Returns:
        list: list of tuples with center
                        indices and radius of circles
    """
    radii = [i[2] for i in props]
    maximum = max(radii)
    props = [i for i in props if i[2] > maximum / 2]

    return props


def circles_inside_circles(props):

    for i in range(len(props)):
        if props[i] != 0:
            x1 = props[i][0] - props[i][2]
            x2 = props[i][0] + props[i][2]
            y1 = props[i][1] - props[i][2]
            y2 = props[i][1] + props[i][2]

        for j in range(len(props)):
            if props[j] != 0 and i != j:
                x_new = props[j][0]
                y_new = props[j][1]
                r_new = props[j][2]

                inside_x = (x1 < x_new - r_new / 2 and x2 > x_new + r_new / 2)
                inside_y = (y1 < y_new -r_new / 2 and y2 > y_new + r_new / 2)

                if inside_x and inside_y:
                    props[j] = 0

    props = [i for i in props if i != 0]

    return props


def filter_circles(roi, circles, area, median):
    """Delete too small and too bright
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
    dims = roi.shape

    i = 0
    while i < len(circles):
        cx = int(circles[i][0])
        cy = int(circles[i][1])
        r = int(circles[i][2])

        pixels = pixels_in_circle(roi, cx, cy, r)
        area_label = sum(pixels)
        area_circle = np.pi * r**2
        outside_mask = area_circle  / 2 > area_label
        bellow_zero = cx - r < 0 or cy - r < 0
        over_size = cx + r > dims[1] or cy + r > dims[0]

        if outside_mask or bellow_zero or over_size:
            circles.remove(circles[i])
        else:
            i += 1

    return circles


def pixels_in_circle(img, cx, cy, r):
    """Find all pixel values inside circle

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
    median = data_list[index]

    if n % 2 == 0:
        value = data_list[index - 1]
        median = (median + value) / 2

    return median


def ploting_img(images, names):
    """Function for plotting images
                    from segmentation proces

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


def saving(img, file_name, sizes, np_type, directory="results"):
    """Save result image into given directory.

    Args:
        img (numpy array): labeled image
        file_name (str): name of current input image file
        sizes (list): list of sizes in current input
        np_type (str): nanoparticles or nanorods
        directory (str, optional): path to directory

    Returns:
        string: path to file
    """
    name = re.split("/", file_name)
    res_path = os.path.join(directory, name[-1])
    plt.imsave(res_path, img)

    size_path = re.split("\\.", res_path)
    size_path = f'{size_path[0]}.txt'

    with open(size_path, "w") as txt_file:
        if np_type == "nanoparticles":
            avg = round(sum(sizes) / len(sizes), 3)
            avg = f"mean diameter: {avg} nm\n"
            txt_file.write(avg)

            for size in sizes:
                curr_size = str(size)
                txt_file.write(curr_size)
                txt_file.write('\n')
        else:
            avg_area = round(sum(sizes[0]) / len(sizes[0]), 3)
            avg_area = "mean area: " + str(avg_area) + "nm^2\n"
            txt_file.write(avg_area)

            avg_major = round(sum(sizes[1]) / len(sizes[1]), 3)
            avg_major = "mean major axis length: " + str(avg_major) + "nm\n"
            txt_file.write(avg_major)

            avg_minor = round(sum(sizes[2]) / len(sizes[2]), 3)
            avg_minor = "mean minor axis length: " + str(avg_minor) + "nm\n"
            txt_file.write(avg_minor)

            header = "area, major_axis, minor_axis\n"
            txt_file.write(header)

            for i in range(len(sizes[0])):
                line = [str(sizes[0][i]), str(sizes[1][i]), str(sizes[2][i])]
                txt_file.writelines(line)
                txt_file.write('\n')

    return res_path, size_path


def calculation_watershed(labeled, np_type):
    """Calculate mean radius and area of NP.

    Args:
        labeled (numpy array): labeled image
        pixel_size (float): size of one pixel in image
        np_type (string): nanoparticles or nanorods

    Returns:
        list, list: list of sizes, list of tuples
                    with label and size
    """
    if np_type == "nanoparticles":
        props = regionprops_table(
            labeled, properties=["label", "area_convex", "equivalent_diameter_area"]
        )
        sizes = props["area_convex"]
        sizes = [2 * (i / np.pi)**(1/2) for i in sizes]      # S = pi * r^2 = pi * d^2 / 4 d = 2 * r = 2 * sqrt(S / pi)

    else:
        props = regionprops_table(
            labeled,
            properties=[
                "label",
                "area_convex",
                "axis_major_length",
                "axis_minor_length",
            ],
        )
        sizes = [
            props["area_convex"],
            props["axis_major_length"],
            props["axis_minor_length"],
        ]

    labels = props["label"]
    area = props["area_convex"]
    selected = [(labels[i], area[i]) for i in range(len(labels))]

    return sizes, selected


def histogram_sizes(sizes, file_name, np_type):
    """Create histogram of sizes of NP

    Args:
        sizes (list): list of NP sizes
    """
    file_name = f'{file_name[:-4]}hist.png'

    if np_type == "nanoparticles":
        plt.hist(sizes, bins=10, color="dodgerblue", edgecolor="black")
        plt.title("Histogram of sizes of NPs")
        plt.xlabel("diameter [nm]")
        plt.ylabel("frequency [-]")
        plt.savefig(file_name)
        plt.clf()

    elif np_type == "nanorods":
        plt.hist(sizes[0], bins=10, color="forestgreen", edgecolor="black", range=[0, 100])
        plt.hist(sizes[1], bins=10, color="brown", edgecolor="black", range=[0, 100])
        plt.legend(["major axis", "minor axis"])
        plt.title("Histogram of sizes of NRs")
        plt.xlabel("axis length [nm]")
        plt.ylabel("frequency [-]")
        plt.savefig(file_name)
        plt.clf()

    return file_name


def read_args():
    """Command line arguments

    Returns:
        dictionary: path to image folder and
                            method of segmentation
    """
    parser = argparse.ArgumentParser(
        description="segmentation of NPs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-p", "--path", type=str, default="data/images", help="path to images (default: data/images)"
    )
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        default="watershed",
        help="segmentation method (default: watershed)",
    )
    parser.add_argument(
        "-s", "--show", type=bool, default=False, help="Plotting image (default: False)"
    )
    args = parser.parse_args()
    config = vars(args)
    print(config)

    return config


def image_analysis(input_description, image, images=None, names=None):
    """Analyze image of NPs and calculate properties

    Args:
        input_description (dict): metadata
        image (path): image file
        images (list, optional): list of already analyzed images. Defaults to None.
        names (list, optional): list of already analyzed images names. Defaults to None.

    Returns:
        path, path, path: path to labeled image, histogram of sizes and txt file with properties
    """

    scale = int(input_description[image][0])
    np_type = input_description[image][1]
    img_path = input_description[image][2]
    if np_type == "nanorods":
        background = True
    else:
        background = False

    img_raw, pixel_size = loading_img(img_path, scale)
    img_filtered, pixel_size = filtering_img(img_raw, scale, np_type, pixel_size, background)
    binary = binarizing(img_filtered, np_type)
    labels, sizes, props_ht = segmentation(
        img_filtered, binary, np_type, pixel_size
    )
    img = result_image(img_raw, labels, np_type, props_ht)

    if images and names:
        images.append(img)
        names.append(img_path)

    labeled_filename, sizes_filename = saving(img, img_path, sizes, np_type)
    hist_filename = histogram_sizes(sizes, labeled_filename, np_type)

    print("saving into:", sizes_filename)

    return labeled_filename, hist_filename, sizes_filename


def get_config():
    """Configuration from command line

    Raises:
        Exception: wrong method

    Returns:
        dict, str, bool: metadata, method, plot result or not
    """

    config = read_args()
    print(config["path"])

    input_description = load_inputs(config["path"])
    method = config["method"]
    show = config["show"]

    if method != "watershed":
        raise Exception("unknown method")

    return input_description, method, show


if __name__ == "__main__":
    input_description, _, show = get_config()

    images = []
    names = []

    for image in input_description:
        image_analysis(input_description, image, images, names)

    if show:
        ploting_img(images, names)
