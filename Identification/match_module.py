import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import dist_module
import histogram_module


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


def get_image_from_path(image_path):
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
    hist_is_gray = histogram_module.is_grayvalue_hist(hist_type)

    model_hists = compute_histograms(model_images, hist_type, hist_is_gray, num_bins)
    query_hists = compute_histograms(query_images, hist_type, hist_is_gray, num_bins)

    best_matches = -np.ones(len(query_images))
    distances_matrix = np.zeros((len(model_images), len(query_images)))

    for query_hist_index, query_hist in enumerate(query_hists):
        min_dist = np.inf

        for model_hist_index, model_hist in enumerate(model_hists):
            curr_dist = dist_module.get_dist_by_name(query_hist, model_hist, dist_type)

            if curr_dist < min_dist:
                min_dist = curr_dist
                best_matches[query_hist_index] = model_hist_index

            distances_matrix[model_hist_index, query_hist_index] = curr_dist

    return best_matches, distances_matrix


def compute_histograms(images_paths_list, hist_type, hist_is_gray, num_bins):
    images_hists = []

    # Compute histogram for each image and add it at the bottom of image_hist
    for img_path in images_paths_list:
        img = get_image_from_path(img_path)

        if hist_is_gray:
            img = rgb2gray(img)

        hist = histogram_module.get_hist_by_name(img, num_bins, hist_type)

        # Handles the case in which the histogram function returns also the bins
        if len(hist) == 2:
            hist = hist[0]

        images_hists.append(hist)

    return images_hists


# For each image file from 'query_images' find and visualize the 5 nearest images from 'model_image'.
#
# Note: use the previously implemented function 'find_best_match'
# Note: use subplot command to show all the images in the same Python figure, one row per query image

def show_neighbors(model_images, query_images, dist_type, hist_type, num_bins):
    num_nearest = 5  # show the top-5 neighbors

    def add_query_image_to_figure(image_path, index):
        axes = figure.add_subplot(subplot_n_rows, subplot_n_cols, subplot_index)
        axes.set_axis_off()
        axes.set_title('Q' + str(index))
        axes.imshow(np.array(Image.open(image_path)))

    def add_model_image_to_figure(image_path, distance):
        axes = figure.add_subplot(subplot_n_rows, subplot_n_cols, subplot_index)
        axes.set_axis_off()
        distance = np.around(distance, decimals=2)
        axes.set_title('M' + str(distance))
        axes.imshow(np.array(Image.open(image_path)))

    def get_top_neighbors(query_image_index):
        distances = distance_matrix[:, query_image_index]
        unsorted_top_neighbors_indexes = np.argpartition(distances, num_nearest)[:num_nearest]
        unsorted_top_neighbors_distances = distances[unsorted_top_neighbors_indexes]

        sorted_args = np.argsort(unsorted_top_neighbors_distances)
        sorted_top_neighbors_indexes = unsorted_top_neighbors_indexes[sorted_args]
        sorted_top_neighbors_distances = unsorted_top_neighbors_distances[sorted_args]

        return sorted_top_neighbors_indexes, sorted_top_neighbors_distances

    # Prepare Plots
    figure = plt.figure(figsize=(10, 10))
    subplot_n_rows = len(query_images)
    subplot_n_cols = num_nearest + 1
    subplot_index = 1

    # Compute images distances
    _, distance_matrix = find_best_match(model_images, query_images, dist_type, hist_type, num_bins)

    for query_image_path_index, query_image_path in enumerate(query_images):
        add_query_image_to_figure(query_image_path, query_image_path_index)
        subplot_index += 1

        top_neighbors_indexes, top_neighbor_distances = get_top_neighbors(query_image_path_index)

        for neighbor_index, neighbor_distance in zip(top_neighbors_indexes, top_neighbor_distances):
            model_image_path = model_images[neighbor_index]
            add_model_image_to_figure(model_image_path, neighbor_distance)
            subplot_index += 1

    plt.show()
