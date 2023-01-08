from skimage import filters, io, color, measure
import numpy as np
import matplotlib.pyplot as plt

def input_image(path):

    raw = io.imread(path)
    gray = color.rgb2gray(raw)
    gray = gray[ : -100, : ]
    filtered = filters.median(gray)

    thresh = filters.threshold_otsu(filtered)
    binary = filtered < thresh

    labels = measure.label(binary, background=0)

    plt.imshow(labels, cmap='nipy_spectral')
    plt.show()



if __name__ == '__main__':
    path = 'images/AuNP20nm_004.jpg'
    input_image(path)