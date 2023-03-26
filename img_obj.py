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
from skimage.morphology import remove_small_holes, remove_small_objects, disk, erosion

class NPImage:
    def __init__(self, img_path, scale, np_type):
        try:
            self.img = imread(img_path)
        except Exception as err:
            print(err)
            raise Exception("wrong input: image cannot be open")
        
        self.scale = scale
        self.np_type = np_type
        self.find_scale()
        self.crop()
        self.prepare()
        self.filter()
        
    def find_scale(self):
        width = self.img.shape[1]
        line = self.img[-100:, int(width / 2) :, :]
        line = cv2.cvtColor(line, cv2.COLOR_RGB2GRAY)
        mask = line < 10
        mask = remove_small_objects(mask)
        indices = np.where(mask)
        length = max(indices[1]) - min(indices[1])
        self.pixel_size = self.scale / length

    def prepare(self):
        self.img = self.img[0:-100, :]
        self.img = rgb2gray(self.img)
        self.img = rescale(self.img, 0.5)
        self.pixel_size *= 2

    def filter(self):
        if self.scale > 500:
            kernel = disk(3)
        elif self.scale > 150:
            kernel = disk(5)
        else:
            kernel = disk(7)
        self.img = median(self.img, kernel)

    def binarize(self):
        if self.np_type == 'nanoparticles':