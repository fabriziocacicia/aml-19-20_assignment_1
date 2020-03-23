from typing import List

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import histogram_module
import dist_module


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


def get_image_from_path(image_path: str) -> np.ndarray:
    """
    Return an image given its path
    :param image_path: The path of the image
    :return: The image
    """
    image = np.array(Image.open(image_path))
    image = image.astype('double')

    return image


# model_images - list of file names of model images
# query_images - list of file names of query images
#
# dist_type - string which specifies distance type:  'chi2', 'l2', 'intersect'
# hist_type - string which specifies histogram type:  'grayvalue', 'dxdy', 'rgb', 'rg'
#
# note: use functions 'get_dist_by_name', 'get_hist_by_name' and 'is_grayvalue_hist' to obtain 
#       handles to distance and histogram functions, and to find out whether histogram function 
#       expects grayvalue or color image

def find_best_match(model_images, query_images, dist_type, hist_type, num_bins):

    hist_isgray = histogram_module.is_grayvalue_hist(hist_type)
    
    model_hists = compute_histograms(model_images, hist_type, hist_isgray, num_bins)
    query_hists = compute_histograms(query_images, hist_type, hist_isgray, num_bins)
    
    D = np.zeros((len(model_images), len(query_images)))
    
    
    #... (your code here)


    return best_match, D



            distances_matrix[query_image_index, model_image_index] = curr_dist

    return best_matches, distances_matrix


def compute_histograms(images_paths_list: List[str], hist_type: str, hist_is_gray: bool, num_bins: int):
    images_hists = []

    # Compute histogram for each image and add it at the bottom of image_hist
    for img_path in images_paths_list:
        img: np.ndarray = get_image_from_path(img_path)

        if hist_is_gray:
            img = rgb2gray(img)

        hist = histogram_module.get_hist_by_name(img, num_bins, hist_type)

        images_hists.append(hist)

    return images_hists


# For each image file from 'query_images' find and visualize the 5 nearest images from 'model_image'.
#
# Note: use the previously implemented function 'find_best_match'
# Note: use subplot command to show all the images in the same Python figure, one row per query image

def show_neighbors(model_images, query_images, dist_type, hist_type, num_bins):
    
    
    plt.figure()

    num_nearest = 5  # show the top-5 neighbors
    
    #... (your code here)

