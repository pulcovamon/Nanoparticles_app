import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from skimage import filters, morphology
from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from skimage.segmentation import watershed
from skimage.transform import rescale
#()


def loading_img(img_path):
    """Function for loading image from given file and croping it

    Args:
        img_path (str): path to image file

    Returns:
        numpy.ndarray: RGB image
    """

    img_raw = imread(img_path)
    img_raw = img_raw[0:-100, :]
    #img_raw = rescale(img_raw, 0.5)

    return img_raw



def filtering_img(img_raw):
    """Function for bluring image and deleting background

    Args:
        img_raw (numpy.ndarray): RGB image

    Returns:
        _numpy.ndarray: grayscale image

    """
    img_raw = rgb2gray(img_raw)
    img_raw = rescale(img_raw, 0.25, anti_aliasing = False)
    img_raw = 1 - img_raw
    img_filtered = filters.median(img_raw, np.ones((7, 7)))
    background = filters.median(img_raw, np.ones((101, 101)))
    img_filtered = img_filtered - background
    plt.imshow(background)
    plt.show()

    return img_filtered



def thresholding_img(img_filtered):
    """Function to binarize image and inverse it

    Args:
        img_filtered (numpy.ndarray): grayscale image

    Returns:
        numpy.ndarray: binary image
    """

    threshold = threshold_otsu(img_filtered)

    img_binary = img_filtered > threshold
    #img_binary = ~img_binary

    return img_binary



def morfological_img(img_binary):
    """Function for using morfological operations erosion and dilation

    Args:
        img_binary (numpy.ndarray): binary image

    Returns:
        numpy.ndarray: binary image
    """

    kernel = morphology.square(width = 5)

    img_erode = morphology.erosion(img_binary, kernel)
    img_dilate = morphology.dilation(img_erode, kernel)

    return img_erode, img_dilate



def segmentation_img(img_dilate, img_binary):
    """Function for segmentation and labeling image

    Args:
        img_dilate (numpy.ndarray): binary image
        img_binary (numpy.ndarray): binary image

    Returns:
        numpy.ndarray: labeled image
    """

    distance = ndi.distance_transform_edt(img_dilate)
    img_mask = distance > 0.5 * np.amax(distance)
    markers, _ = ndi.label(img_mask)

    img_segment = watershed(-distance, markers, mask = img_binary)

    return img_segment



def ploting_img(img_raw, img_binary, img_dilate, img_segment):
    """Function for plotting images from segmentation proces

    Args:
        img_raw (numpy.ndarray): RGB image
        img_binary (numpy.ndarray): binary image
        img_dilate (numpy.ndarray): binary image
        img_segment (numpy.ndarray): labeled image
    """

    plt.subplot(2, 2, 1)
    plt.imshow(img_raw)
    plt.title('Raw image')

    plt.subplot(2, 2, 2)
    plt.imshow(img_binary, cmap = 'gray')
    plt.title('Binary image')

    plt.subplot(2, 2, 3)
    plt.imshow(img_dilate, cmap = 'gray')
    plt.title('Image after morfological operations')

    plt.subplot(2, 2, 4)
    plt.imshow(img_segment, cmap = 'tab20')
    plt.title('Segmented image')

    plt.show()



if __name__ == '__main__':
    img_path = r'C:\Users\drakv\Desktop\fbmi\projekt\projekt\app\AuNP20nm_004.jpg'

    img_raw = loading_img(img_path)

    img_filtered = filtering_img(img_raw)

    img_binary = thresholding_img(img_filtered)

    img_erode, img_dilate = morfological_img(img_binary)

    img_segment = segmentation_img(img_dilate, img_binary)

    ploting_img(img_raw, img_binary, img_dilate, img_segment)
