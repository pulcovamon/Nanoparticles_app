import cv2
import numpy as np
import matplotlib.pyplot as plt


def loading_img(img_path):
    """Function for reading image from given file and crop it

    Args:
        img_path (str): path to image file

    Returns:
        numpy.ndarray: BGR image
    """

    img_raw = cv2.imread(img_path)
    img_raw = img_raw[0:-100, :]

    return img_raw


def filtering_img(img_raw):
    """Funtion for filtering image with median filter and deleting background

    Args:
        img_raw (numpy.ndarray): BGR image

    Returns:
        numpy.ndarray: grayscale image
    """

    img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
    bck = cv2.GaussianBlur(img_raw, (169, 169), 0)
    img_filtered = cv2.medianBlur(img_raw, 5)

    #bcg_substractor = cv2.createBackgroundSubtractorMOG2(detectShadows = False)
    #fmask = bcg_substractor.apply(img)

    return img_filtered


def thresholding_img(img_filtered):
    """Function to binarize image using Otsu threshold

    Args:
        img_filtered (numpy.ndarray): grayscale image

    Returns:
        numpy.ndarray: binary image
    """

    _, img_binary = cv2.threshold(img_filtered, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(type(img_binary))

    return img_binary


def morfological_img(img_binary):
    """Function for making sure background and sure foreground of image using morfological operations erosion and dilation

    Args:
        img_binary (numpy.ndarray): binary image

    Returns:
        numpy.ndarray: binary image
    """

    img_binary = 255-img_binary

    frg_kernel = np.ones((9, 9), np.uint8)
    bcg_kernel = np.ones((5, 5), np.uint8)

    bcg = cv2.dilate(img_binary, bcg_kernel, iterations=1)
    frg = cv2.erode(img_binary, frg_kernel, iterations=6)

    return bcg, frg


def distance_image(frg, bcg):
    """Function for calculating distance map

    Args:
        frg (numpy.ndarray): binary image
        bcg (numpy.ndarray): binary image

    Returns:
        numpy.ndarray: distance map
    """

    distance_map = cv2.distanceTransform(frg, distanceType=cv2.DIST_C, maskSize=5)
    _, frg = cv2.threshold(frg, 0.7 * distance_map.max(), 255, 0)
    frg = np.uint8(frg)
    particles = cv2.subtract(bcg, frg)

    return particles, distance_map


def segmentation_img(frg, particles, img_raw):
    """Function for segmentation and labeling image using watershed transform

    Args:
        frg (numpy.ndarray): binary image
        particles (numpy.ndarray): binary image
        img_raw (numpy.ndarray): BGR image

    Returns:
        numpy.ndarray: labeled image
    """

    _, labels = cv2.connectedComponents(frg)
    labels = labels + 1
    labels[particles == 255] = 0

    labels = cv2.watershed(img_raw, labels)

    return labels


def ploting_img(img_raw, img_binary, distance_map, labels):
    """Function for plotting images from segmentation proces

    Args:
        img_raw (numpy.ndarray): BGR image
        img_binary (numpy.ndarray): binary image
        distance_map (numpy.ndarray): distance map
        labels (numpy.ndarray): labeled image
    """

    plt.subplot(2, 2, 1)
    plt.imshow(img_raw)
    plt.title('Raw image')

    plt.subplot(2, 2, 2)
    plt.imshow(img_binary, cmap = 'gray')
    plt.title('Binary image')

    plt.subplot(2, 2, 3)
    plt.imshow(distance_map, cmap = 'gray')
    plt.title('Distance map')

    plt.subplot(2, 2, 4)
    plt.imshow(labels, cmap = 'tab20')
    plt.title('Segmented image')

    plt.show()


if __name__ == '__main__':
    img_path = r'C:\Users\drakv\Desktop\fbmi\projekt\projekt\app\AuNP20nm_004.jpg'

    img_raw = loading_img(img_path)

    img_filtered = filtering_img(img_raw)

    img_binary = thresholding_img(img_filtered)

    bcg, frg = morfological_img(img_binary)

    particles, distance_map = distance_image(frg, bcg)

    labels = segmentation_img(frg, particles, img_raw)

    ploting_img(img_raw, img_binary, distance_map, labels)
