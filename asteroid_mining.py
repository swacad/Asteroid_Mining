import numpy as np
import cv2
from scipy import spatial
import matplotlib.pyplot as plt

import os
import time


def get_hues(image):
    """
    Converts image to HSV format and slices out the hue channel normalized from 0 to 360.
    Parameters
    ----------
    image: np.array as 3-channel BGR image of uint8 dtype

    Returns
    -------
    hues: np.array of float32 dtype

    """
    hues = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Slice out hue channel as float32 array
    hues = np.array(hues[:, :, 0], dtype='float32')

    # Normalize hues to 0 to 359 degrees
    hues = hues * 2

    return hues


def compute_wavelengths(image):
    """
    Computes approximate wavelength in microns for a given image.
    NOTE: Wavelengths are only approximations as it is impossible to convert RGB or HSV values back to the original
    wavelength values without additional data.
    Parameters
    ----------
    hues: np.array of dtype float32 with values in the range of 0 to 359

    Returns
    -------
    wavelengths: np.array of dtype float32
    """
    hues = get_hues(image)
    # Estimating that the usable part of the visible spectrum is 450-620nm,
    # with wavelength (in microns) and hue value (in degrees)
    wavelengths = 620 - 170 / 270 * hues

    # Divide by 1000 to normalize to microns
    wavelengths = wavelengths / 1000
    # print(wavelengths)

    # Apply Gaussian blur to filter out noise on image data
    kernel_size = (5, 5)
    wavelengths = cv2.GaussianBlur(wavelengths, kernel_size, 0)

    return wavelengths


def plot_histogram(wavelengths, filename):
    plot_min = np.floor(np.around(wavelengths.min() * 1000, 3)) / 1000
    plot_max = np.ceil(np.around(wavelengths.max() * 1000, 3)) / 1000
    hist = cv2.calcHist([wavelengths], [0], None, [100], [plot_min, plot_max]).flatten()
    plt.plot(hist)
    plt.title('Wavelengths Histogram')
    plt.ylabel('Count (log scale)')
    plt.yscale('log')
    plt.xlabel('100 Bins range: ' + str(plot_min) + ' - ' + str(plot_max) + ' microns')
    # plt.show()
    plt.savefig(filename)



def print_wavelength_info(wavelengths):
    print('wavelengths.min() = ' + str(wavelengths.min()))
    print('wavelengths.max() = ' + str(wavelengths.max()))
    print('wavelengths.mean() = ' + str(wavelengths.mean()))
    print('wavelengths.std() = ' + str(wavelengths.std()))
    print('')


def main():
    # Driver code
    green = (0, 255, 0)
    ida = cv2.imread("images/ida.jpg")
    patch = ida[500:580, 450:575]
    cv2.rectangle(ida, (450, 500), (575, 580), green)
    # cv2.imshow('patch', patch)  # [debug]
    # cv2.waitKey(0)
    # cv2.imwrite('patch1.jpg', patch)
    wavelengths = compute_wavelengths(patch)
    print_wavelength_info(wavelengths)
    plot_histogram(wavelengths, 'images/ida_hist.png')
    cv2.imwrite('images/ida_rect.png', ida)

    olivine = cv2.imread('images/Lunar_Olivine_Basalt_15555_from_Apollo_15.jpg')
    patch = olivine[850:1550, 1250:1950]
    cv2.rectangle(olivine, (1250, 850), (1950, 1550), green)
    wavelengths = compute_wavelengths(patch)
    print_wavelength_info(wavelengths)
    plot_histogram(wavelengths, 'images/olivine_hist.png')
    cv2.imwrite('images/olivine_rect.png', olivine)

    chondrite = cv2.imread('images/chondrite (contains olivine and pyroxene).jpg')
    patch = chondrite[200:500, 200:600]
    cv2.rectangle(chondrite, (200, 200), (600, 500), green)
    wavelengths = compute_wavelengths(patch)
    print_wavelength_info(wavelengths)
    plot_histogram(wavelengths, 'images/chondrite_hist.png')
    cv2.imwrite('images/chondrite_rect.png', chondrite)

    eros = cv2.imread('images/eros_asteroid.tif')
    patch = eros[315:355, 470:550]
    cv2.rectangle(eros, (470, 315), (550, 355), green)
    wavelengths = compute_wavelengths(patch)
    print_wavelength_info(wavelengths)
    plot_histogram(wavelengths, 'images/eros_hist.png')
    cv2.imwrite('images/eros_rect.png', eros)


if __name__ == "__main__":
    main()
