from scipy.spatial import distance
from scipy import stats
import numpy as np
import os
from PIL import Image


def distance_measuring(dataset_path, features, query_images_features):

    all_sorted_dists_for_all_quary_images = []
    all_accepted_images_for_all_quary_images = []

    metric = int(input('Metric number'))

    for each_query_image_features in query_images_features :
        all_dists = dict()
        k = 0

        for image_name in os.listdir(dataset_path):

            if metric == 1:  # Manhattan
                # dist = np.sum(np.absolute(features[k] - each_query_image_features))
                dist = np.sum((np.absolute(features[k] - each_query_image_features)) /
                              (1 + features[k] - each_query_image_features))

            elif metric == 2:  # euclidean
                dist = distance.euclidean(features[k], each_query_image_features)

            elif metric == 3:  # standardized euclidean
                weights = np.ones(shape=features[k, :].shape)
                dist = distance.seuclidean(features[k], each_query_image_features, weights)

            elif metric == 4:  # Mahalanobis
                t = np.zeros(shape=1000)
                # flag = np.zeros(shape=1000)
                try:
                    weights = np.cov(features)
                    t = np.linalg.cholesky(weights)

                except ValueError:
                    print('The matrix is not positive semidefinite. Please choose another similarity metric!')

                t_inverse = np.linalg.inv(t)
                dist = distance.mahalanobis(features[k], each_query_image_features, t_inverse)

            elif metric == 5:  # City block
                dist = distance.cityblock(features[k], each_query_image_features)

            elif metric == 6:  # Minkowski
                dist = distance.minkowski(features[k], each_query_image_features)

            elif metric == 7:  # Chebyshev
                dist = distance.chebyshev(features[k], each_query_image_features)

            elif metric == 8:  # Cosine
                dist = distance.cosine(features[k], each_query_image_features)

            elif metric == 9:  # Correlation
                dist = distance.correlation(features[k], each_query_image_features)

            elif metric == 10:  # Spearman
                dist = stats.spearmanr(features[k], each_query_image_features)

            all_dists[image_name] = dist
            k += 1

        sorted_distances = sorted(all_dists.items(), key=lambda x: x[1], reverse=False)

        accepted_images = []
        for i in range(0, 10):
            image_name = np.array(sorted_distances[i])[0]
            image = Image.open(os.path.join(dataset_path, image_name))
            image = image.resize((224, 224))
            image = np.array(image)
            accepted_images.append(image)

        all_sorted_dists_for_all_quary_images.append(sorted_distances)
        all_accepted_images_for_all_quary_images.append(accepted_images)

    return all_sorted_dists_for_all_quary_images, all_accepted_images_for_all_quary_images
