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
        self.prepare()
        self.filter()
        self.binary = None
        self.props_wsh = []
        self.sizes = []
        self.labels = None
        self.props_ht = None

        
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
        if self.np_type == "nanoparticles":
            thresh = threshold_minimum(self.img)
        else:
            self.thresh = threshold_otsu(self.img)
        self.binary = self.img < thresh
        self.binary = remove_small_holes(self.binary)
        self.binary = remove_small_objects(self.binary)


    def segmentation(self):
        self.binarize()
        print(self.binary.shape)
        self.watershed_transform()
        self.props_wsh = self.calc_watershed(self.labels)

        if self.np_type == "nanoparticles":
            self.find_overlaps()
            self.find_duplicity()
        else:
            self.sizes[0] = [i * self.pixel_size for i in self.sizes[1]]
            self.sizes[1] = [i * self.pixel_size for i in self.sizes[2]]
            props_ht = []


    def watershed_transform(self):
        distance = ndi.distance_transform_edt(self.binary)
        img = -distance
        
        if self.np_type == 'nanoparticles':
            coords = peak_local_max(distance, footprint=np.ones((3, 3)), min_distance=20)
            mask = np.zeros(distance.shape, dtype=bool)
            mask[tuple(coords.T)] = True
        else:
            mask =  self.binary
            for _ in range(10):
                mask = remove_small_holes(mask)
                mask = remove_small_objects(mask)
            for _ in range(10):
                mask = erosion(mask)

        markers, _ = ndi.label(mask)
        self.labels = watershed(img, markers, mask=self.binary)


    def hough_transform(self):
        canny_edge = canny(self.img, sigma=2)
        start = 10 / self.pixel_size
        end = 100 / self.pixel_size
        min_dist = 10/self.pixel_size
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
    
    
    def calc_watershed(self, labeled):
        if self.np_type == "nanoparticles":
            props = regionprops_table(
                labeled, properties=[
                    "label",
                    "area_convex",
                    "equivalent_diameter_area"
                ]
            )
            self.sizes = props["area_convex"]

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
            self.sizes = [
                props["area_convex"],
                props["axis_major_length"],
                props["axis_minor_length"],
            ]

        labels = props["label"]
        area = props["area_convex"]
        selected = [(labels[i], area[i]) for i in range(len(labels))]

        return selected


    def calc_median(self, data_list):
        """Method for calculating median

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
    
    
    def inside_circles(self, props):
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
    

    def pixels_in_circle(self, roi, cx, cy, r):
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

        for x in range(roi.shape[1]):
            for y in range(roi.shape[0]):
                dx = x - cx
                dy = y - cy
                distance = dx**2 + dy**2

                if distance <= r**2:
                    pixels.append(roi[y, x])

        return pixels
        
    
    def filter_circles(self, roi, circles, area):
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
        circles = self.inside_circles(circles)

        dims = roi.shape
        thresh = threshold_minimum(roi)

        for i in range(len(circles)):
            cx = int(circles[i][0])
            cy = int(circles[i][1])
            r = int(circles[i][2])
            pixels = self.pixels_in_circle(roi, cx, cy, r)
            if pixels:
                median_value = self.calc_median(pixels)
                bellow_zero = cx - r < 0 or cy - r < 0
                over_size = cx + r > dims[1] or cy + r > dims[0]
                if median_value > thresh:
                    circles.pop(i)
                    area.pop(i)
                    i += 1
                elif bellow_zero or over_size:
                    circles.pop(i)
                    area.pop(i)
                    i += 1

    
    def find_overlaps(self):
        erode = erosion(self.binary)
        iterations = int(5/self.pixel_size)
        for _ in range(iterations):
            erode = erosion(erode)
        labels = label(erode, background=0)
        sizes_watershed = self.calc_watershed(labels)

        median = self.calc_median(self.sizes)
        props_ht = []

        for number, size in sizes_watershed:
            if size > 1.5 * median:
                indices = np.argwhere(labels == number)
                x_min = max(np.min(indices[:, 0]) - 10, 0)
                y_min = max(np.min(indices[:, 1]) - 10, 0)
                w = labels.shape[0]
                h = labels.shape[1]
                x_max = min(np.max(indices[:, 0]) + 10, w - 1)
                y_max = min(np.max(indices[:, 1]) + 10, h - 1)
                roi = self.img[x_min:x_max, y_min:y_max]

                circles = self.hough_transform()
                area = [(r * self.pixel_size) ** 2 * np.pi for _, _, r in circles]
                circles = self.filter_circles(roi, circles, area)

                for circle in circles:
                    cx = circle[0] + y_min
                    cy = circle[1] + x_min
                    r = circle[2]
                    self.props_ht.append((cx, cy, r))


    def find_duplicity(self, labels):
        """Eliminate duplicities in properties

        Args:
            labels (numpy.ndarray):
            props_ht (_type_): _description_
            props_watershed (_type_): _description_

        Returns:
            _type_: _description_
        """
        size_ht = []
        for cx, cy, r in self.props_ht:
            val = labels[cy, cx]
            props_wsh = [i for i in props_wsh if i[0] != val]
            d = 2 * r
            size_ht.append(d)

        diameter = lambda a: 2 * (a / np.pi) ** (1 / 2)
        size_wsh = [diameter(i[1]) for i in props_wsh]
        sizes = size_ht + size_wsh
        sizes = [round(self.pixel_size * i, 3) for i in sizes]

        return sizes


if __name__ == '__main__':
    image = NPImage(
        'data/images/AuNP_20nm_001.jpg',
        100, 'nanoparticles')
    
    image.segmentation()