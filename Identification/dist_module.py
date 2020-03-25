import numpy as np


# Compute the intersection distance between histograms x and y
# Return 1 - hist_intersection, so smaller values correspond to more similar histograms
# Check that the distance range in [0,1]

def dist_intersect(x, y):
    assert len(x) == len(y), "Histogram sizes mismatch"

    sum_of_mins = np.sum(np.minimum(x, y))

    hist_intersection = 0.5 * (sum_of_mins / np.sum(x) + sum_of_mins / np.sum(y))

    distance = 1 - hist_intersection

    assert 0 < distance < 1, "Distance out of range [0, 1]"

    return distance


# Compute the L2 distance between x and y histograms
# Check that the distance range in [0,sqrt(2)]

def dist_l2(x, y):
    assert len(x) == len(y), "Histogram sizes mismatch"

    distance = np.sum(np.square(np.subtract(x, y)))

    assert 0 < distance < np.sqrt(2), "Distance value out of range"

    return distance


# Compute chi2 distance between x and y
# Check that the distance range in [0,Inf]
# Add a minimum score to each cell of the histograms (e.g. 1) to avoid division by 0

def dist_chi2(x, y):
    assert len(x) == len(y), "Histogram sizes mismatch"
    x_ = x + 1
    y_ = y + 1

    diff = np.subtract(x_, y_)

    sq_diff = diff ** 2

    distance = np.sum(np.divide(sq_diff, np.add(x_, y_)))

    assert 0 < distance < np.inf, "Distance value out of range"

    return distance


def get_dist_by_name(x, y, dist_name):
    if dist_name == 'chi2':
        return dist_chi2(x, y)
    elif dist_name == 'intersect':
        return dist_intersect(x, y)
    elif dist_name == 'l2':
        return dist_l2(x, y)
    else:
        assert False, 'unknown distance: %s' % dist_name
