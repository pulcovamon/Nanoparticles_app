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
from skimage.segmentation import watershed
from skimage.feature import peak_local_max, canny
from skimage.transform import rescale, hough_circle, hough_circle_peaks
from skimage.measure import regionprops_table, label
from skimage.morphology import remove_small_holes, remove_small_objects, disk


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
    pixel_size = scale / length

    img_raw = img_raw[0:-100, :]

    return img_raw, pixel_size


def filtering_img(img_raw, scale, np_type, pixel_size):
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
    ...


def binarizing(img, np_type):
    """Find threshold and binarize image

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
    plt.imshow(binary)
    #plt.show()
    binary = remove_small_holes(binary)
    binary = remove_small_objects(binary)
    plt.imshow(binary)
    #plt.show()

    return binary


def watershed_transform(binary):
    """Label image using calculated
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
    labels = watershed_transform(binary)
    sizes, props_watershed = calculation_watershed(labels, np_type)
    labels, props_watershed = filter_blobs(labels, sizes, props_watershed)

    if np_type == "nanoparticles":
        props_ht = find_overlaps(img, binary, sizes, np_type, pixel_size)
        sizes = find_duplicity(labels, props_ht, props_watershed, pixel_size)
    else:
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
    multiplicator = int(255 / max_val)
    labels = np.uint8(labels * multiplicator)
    color_map = plt.get_cmap("nipy_spectral")
    labels = color_map(labels)
    img = overlay_images(raw, labels)

    if np_type == "nanoparticles":
        for x, y, r in props_ht:
            x *= 2
            y *= 2
            r *= 2
            cv2.circle(img, (x, y), r, (255, 0, 255), 2)

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
    img = (7 * raw + 3 * rgb) / 10
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


def find_overlaps(img, binary, sizes, np_type, pixel_size):
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
    labels = label(binary, background=0)
    _, props = calculation_watershed(labels, np_type)

    median = calc_median(deepcopy(sizes))
    props_ht = []

    for number, size in props:
        if size > 1.5 * median:
            indices = np.argwhere(labels == number)
            x_min = max(np.min(indices[:, 0]) - 10, 0)
            y_min = max(np.min(indices[:, 1]) - 10, 0)
            w = labels.shape[0]
            h = labels.shape[1]
            x_max = min(np.max(indices[:, 0]) + 10, w - 1)
            y_max = min(np.max(indices[:, 1]) + 10, h - 1)
            roi = img[x_min:x_max, y_min:y_max]

            circles = hough_segmentation(roi, pixel_size, np_type)
            area = [(r * pixel_size) ** 2 * np.pi for x, y, r in circles]
            circles = filter_circles(roi, circles, area)

            for circle in circles:
                cx = circle[0] + y_min
                cy = circle[1] + x_min
                r = circle[2]
                props_ht.append((cx, cy, r))

    return props_ht


def filter_blobs(labels, sizes, props):
    """Eliminate oversegmented blobs

    Args:
        labels (numpy.ndarray): labeled image from watershed
        sizes (list): list of sizes from watershed transform
        props (list): list of tuples with number of label
                        and particular size from watershed

    Returns:
        numpy.ndarray, list: labeled image, list of properties
    

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
            i += 1"""   


    return labels, props


def hough_segmentation(img, pixel_size, np_type):
    """Segmentation using hough circle and elipse transform

    Args:
        img_filtered (numpy.ndarray): grayscale image

    Returns:
        numpy.ndarray, list: RGB image with plotted
                circles/elipses, list of NP parameters
    """
    canny_edge = canny(img, sigma=2)

    start = max(10, 5 / pixel_size)
    end = min(100, 50 / pixel_size)
    min_dist = 10/pixel_size
    hough_radii = np.arange(start, end, 2)
    hough_res = hough_circle(canny_edge, hough_radii)

    accums, x, y, r = hough_circle_peaks(
        hough_res, hough_radii, min_xdistance=int(min_dist), min_ydistance=int(min_dist)
    )
    x = np.uint16(np.around(x))
    y = np.uint16(np.around(y))
    r = np.uint16(np.around(r))

    a = accums
    l = len(x)
    circles = [(x[i], y[i], r[i]) for i in range(l) if a[i] > 0.25]

    return circles

def inside_circles(props):

    for x, y, r in props:
        x1 = x - r
        x2 = x + r
        y1 = y - r
        y2 = y + r

        for x_new, y_new, r_new in props:
            x1_new = x_new - r_new
            x2_new = x_new + r_new
            y1_new = y_new - r_new
            y2_new = y_new + r_new

            inside_x = (x1 < x1_new and x2 > x2_new)
            inside_y = (y1 < y1_new and y2 > y2_new)

            if inside_x and inside_y:
                x_new = 0
                y_new  = 0
                r_new = 0

    props = [i for i in props if i[2] != 0]

    return props


def filter_circles(gray, circles, area):
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
    print('before: ', circles)
    circles = inside_circles(circles)
    print('after: ', circles)

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
        max_area = max(median_area, mean_area)

        if first_cycle:
            pixels = pixels_in_circle(gray, cx, cy, r)
            median_value = calc_median(copy(pixels))
            bellow_zero = cx - r < 0 or cy - r < 0
            over_size = cx + r > dims[1] or cy + r > dims[0]
            if median_value > thresh:
                circles.pop(i)
                area.pop(i)
            elif bellow_zero or over_size:
                circles.pop(i)
                area.pop(i)
            else:
                i += 1
        else:
            smaller = area[i] <= max_area / 1.5
            bigger = area[i] >= max_area * 3
            '''if smaller or bigger:
                circles.pop(i)
                area.pop(i)
                j = 0
            else:'''
            i += 1
            j += 1

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
    file_name = f'{file_name[:-4]}hist{file_name[-4:]}'

    if np_type == "nanoparticles":
        plt.hist(sizes, bins=10, color="brown", edgecolor="black")
        plt.title("Histogram of sizes of NPs")
        plt.xlabel("diameter [nm]")
        plt.ylabel("frequency")
        plt.savefig(file_name)
        plt.clf()

    elif np_type == "nanorods":
        plt.hist(sizes[0], bins=10, color="forestgreen", edgecolor="black")
        plt.hist(sizes[1], bins=10, color="brown", edgecolor="black")
        plt.legend(["major axis", "minor axis"])
        plt.title("Histogram of sizes of NRs")
        plt.xlabel("axis length [nm]")
        plt.ylabel("frequency")
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

    img_raw, pixel_size = loading_img(img_path, scale)
    img_filtered, pixel_size = filtering_img(img_raw, scale, np_type, pixel_size)
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
