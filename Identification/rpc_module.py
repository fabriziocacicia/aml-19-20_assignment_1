import numpy as np
import matplotlib.pyplot as plt

import histogram_module
import dist_module
import match_module


# compute and plot the recall/precision curve
#
# D - square matrix, D(i, j) = distance between model image i, and query image j
#
# note: assume that query and model images are in the same order, i.e. correct answer for i-th query image is the i-th model image


def plot_rpc(distance_matrix, plot_color):
    recalls = []
    precisions = []

    num_model_images = distance_matrix.shape[0]
    num_queries_images = distance_matrix.shape[1]

    assert (num_model_images == num_queries_images), 'Distance matrix should be a square matrix'

    labels = np.diag([1] * num_model_images)

    flatten_distance_matrix = distance_matrix.reshape(distance_matrix.size)
    flatten_labels = labels.reshape(labels.size)

    sorted_distance_matrix_indexes = flatten_distance_matrix.argsort()
    sorted_flat_distance_matrix = flatten_distance_matrix[sorted_distance_matrix_indexes]
    sorted_flat_labels = flatten_labels[sorted_distance_matrix_indexes]

    num_true_positives: float = 0.0
    num_false_positives: float = 0.0
    num_false_negatives: float = 0.0

    for idt in range(len(sorted_flat_distance_matrix)):
        current_label = sorted_flat_labels[idt]

        num_true_positives += current_label

        # Compute precision and recall values and append them to "recall" and "precision" vectors
        precision = num_true_positives / (idt + 1)
        precisions.append(precision)

        recall = num_true_positives / num_queries_images
        recalls.append(recall)

    plt.plot([1 - precisions[i] for i in range(len(precisions))], recalls, plot_color + '-')


def compare_dist_rpc(model_images, query_images, dist_types, hist_type, num_bins, plot_colors):
    assert len(plot_colors) == len(dist_types), 'number of distance types should match the requested plot colors'

    for idx in range(len(dist_types)):
        [_, distance_matrix] = match_module.find_best_match(model_images, query_images, dist_types[idx], hist_type,
                                                            num_bins)

        plot_rpc(distance_matrix, plot_colors[idx])

    plt.axis([0, 1, 0, 1])
    plt.xlabel('1 - precision')
    plt.ylabel('recall')

    # legend(dist_types, 'Location', 'Best')

    plt.legend(dist_types, loc='best')
