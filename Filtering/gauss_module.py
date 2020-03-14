# import packages: numpy, math (you might need pi for gaussian functions)
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import convolve2d as conv2


def gauss(sigma):
    """
    Gaussian function taking as argument the standard deviation sigma
    The filter should be defined for all integer values x in the range [-3sigma,3sigma]
    The function should return the Gaussian values Gx computed at the indexes x
    :param sigma: the variance of the Gaussian
    :return: gx: the values of the Gaussian, x: the values where the Gaussian is defined
    """
    sigma = round(sigma)

    x = np.arange(-3 * sigma, 3 * sigma + 1, 1)

    x_sq = np.square(x)
    sigma_sq = np.square(sigma)
    exp = np.exp(-x_sq / (2 * sigma_sq))
    
    gx = (1 / (np.sqrt(2 * np.pi) * sigma)) * exp
    
    return gx, x



"""
Implement a 2D Gaussian filter, leveraging the previous gauss.
Implement the filter from scratch or leverage the convolve2D method (scipy.signal)
Leverage the separability of Gaussian filtering
Input: image, sigma (standard deviation)
Output: smoothed image
"""
def gaussianfilter(img, sigma):
    
    #...

    return smooth_img



"""
Gaussian derivative function taking as argument the standard deviation sigma
The filter should be defined for all integer values x in the range [-3sigma,3sigma]
The function should return the Gaussian derivative values Dx computed at the indexes x
"""
def gaussdx(sigma):

    #...
    
    return Dx, x



def gaussderiv(img, sigma):

    #...
    
    return imgDx, imgDy

