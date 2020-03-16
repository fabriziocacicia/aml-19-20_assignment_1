# import packages: numpy, math (you might need pi for gaussian functions)
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import convolve2d as conv2


def gauss(sigma: float):
    """
    Gaussian function taking as argument the standard deviation sigma
    The filter should be defined for all integer values x in the range [-3sigma,3sigma]
    The function should return the Gaussian values Gx computed at the indexes x
    :param sigma: the variance of the Gaussian
    :return: [gx: the values of the Gaussian, x: the values where the Gaussian is defined]
    """
    sigma = round(sigma)

    x = np.arange(-3 * sigma, 3 * sigma + 1, 1)

    x_sq = np.square(x)
    sigma_sq = np.square(sigma)
    exp = np.exp(-x_sq / (2 * sigma_sq))

    gx = (1 / (np.sqrt(2 * np.pi) * sigma)) * exp
    
    return gx, x


def gaussianfilter(img, sigma: float):
    """
    Implement a 2D Gaussian filter, leveraging the previous gauss.
    Implement the filter from scratch or leverage the convolve2D method (scipy.signal)
    Leverage the separability of Gaussian filtering
    :param sigma: The standard deviation of the gaussian
    :param img:
    :return: smooth_img: the smoothed image
    """
    #...
    gx, x = gauss(sigma)

    horizontal_kernel = np.reshape(gx, (1, x.size))
    vertical_kernel = np.reshape(gx, (x.size, 1))

    h_conv = conv2(img, horizontal_kernel, mode='same', boundary='fill')

    v_conv = conv2(h_conv, vertical_kernel, mode='same', boundary='fill')

    smooth_img = v_conv

    return smooth_img


def gaussdx(sigma: float):
    """
    Gaussian derivative function taking as argument the standard deviation sigma
    The filter should be defined for all integer values x in the range [-3sigma,3sigma]
    The function should return the Gaussian derivative values Dx computed at the indexes x
    :param sigma: The standard deviation of the gaussian
    :return: dx: the values of the derivative of the Gaussian, x: the values where the Gaussian is defined
    """

    gx, x = gauss(sigma)

    sigma_sq = np.square(sigma)
    dx = -gx*x/sigma_sq

    return dx, x


def gaussderiv(img, sigma: float):

    dx, x = gaussdx(sigma)

    horizontal_kernel = np.reshape(dx, (1, x.size))
    vertical_kernel = np.reshape(dx, (x.size, 1))

    imgDx = conv2(img, horizontal_kernel, mode='same', boundary='fill')

    imgDy = conv2(img, vertical_kernel, mode='same', boundary='fill')
    
    return imgDx, imgDy

