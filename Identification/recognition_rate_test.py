from typing import List
import match_module
import numpy as np


def get_args():
    """
    Gets the arguments from the command line.
    """
    import argparse
    parser = argparse.ArgumentParser("Compute recognition rates")

    # Args descriptions
    dists_desc = "The list of distances to try ('l2, 'intersect', 'chi2')"
    hists_desc = "The list of histograms to try ('grayvalue', 'rgb', 'rg', 'dxdy')"
    n_bins_desc = "The list of number of bins try (5 10 20 30)"

    # Add the arguments
    parser.add_argument("-dists", help=dists_desc, nargs="+", default=['l2', 'intersect', 'chi2'], type=str)
    parser.add_argument("-hists", help=hists_desc, nargs="+", default=['grayvalue', 'rgb', 'rg', 'dxdy'], type=str)
    parser.add_argument("-bins", help=n_bins_desc, nargs="+", default=[5, 10, 20, 40], type=int)

    args = parser.parse_args()

    return args


def get_images_paths(txt_filename: str) -> List[str]:
    with open(txt_filename) as fp:
        images_paths = fp.readlines()

    images_paths = [x.strip() for x in images_paths]

    return images_paths


def get_model_images_paths() -> List[str]:
    return get_images_paths('model.txt')


def get_query_images_paths():
    return get_images_paths('query.txt')


def compute_recognition_rate(dist_type, hist_type, num_bins, model_images_paths, query_images_paths):
    best_matches, _ = match_module.find_best_match(model_images_paths, query_images_paths, dist_type, hist_type,
                                                   num_bins)
    num_query_images = len(query_images_paths)
    num_correct_matches = sum(best_matches == range(num_query_images))
    recognition_rate = num_correct_matches / num_query_images
    print("\n\tRecognition rate for: dist_type=%s, hist_types=%s, num_bins=%d"
          "\n\t%d" % (dist_type, hist_type, num_bins, recognition_rate))

    return recognition_rate, num_correct_matches, (dist_type, hist_type, num_bins)


def main():
    args = get_args()

    dists_types: List[str] = args.dists
    hists_types: List[str] = args.hists
    list_num_bins: List[int] = args.bins

    num_correct_matches: int = 0
    best_recognition_rate: float = -1.0
    best_combination = ['Nan', 'Nan', -1]

    model_images_paths: List[str] = get_model_images_paths()
    query_images_paths: List[str] = get_query_images_paths()

    from concurrent import futures
    ex = futures.ThreadPoolExecutor(max_workers=10)

    wait_for = [
        ex.submit(compute_recognition_rate, dist_type, hist_type, num_bins, model_images_paths, query_images_paths)
        for dist_type in dists_types for hist_type in hists_types for num_bins in list_num_bins
    ]

    results = []

    for f in futures.as_completed(wait_for):
        result = f.result()
        print('main: result: {}'.format(result))
        results.append(result)

    print(results)

    sorted_results = np.sort(results, axis=0)

    print(sorted_results)

    # print("Best combination: ",
    #      best_combination)  # with dist_type=%s, hist_types=%s, num_bins=%d" % (dist, hist, bins))
    # print('number of correct matches: %d (%f)\n' % (num_correct_matches, best_recognition_rate))


if __name__ == '__main__':
    main()
