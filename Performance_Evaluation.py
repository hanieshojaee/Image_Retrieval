import os
import numpy as np


def performance_evaluation(query_images_names, all_sorted_dists_for_all_quary_images, total_retrieved_images,
                           total_relevant_retrieved_images):

    total_retrieved_images = int(total_retrieved_images)
    total_relevant_retrieved_images = int(total_relevant_retrieved_images)
    precision = []
    recall = []

    for query_image_name in query_images_names :
        query_image_name = int((os.path.splitext(query_image_name))[0])

        img = 0
        if query_image_name in np.arange(0, 100):
            query_image_category = 0
        elif query_image_name in np.arange(100, 200):
            query_image_category = 1
        elif query_image_name in np.arange(200, 300):
            query_image_category = 2
        elif query_image_name in np.arange(300, 400):
            query_image_category = 3
        elif query_image_name in np.arange(400, 500):
            query_image_category = 4
        elif query_image_name in np.arange(500, 600):
            query_image_category = 5
        elif query_image_name in np.arange(600, 700):
            query_image_category = 6
        elif query_image_name in np.arange(700, 800):
            query_image_category = 7
        elif query_image_name in np.arange(800, 900):
            query_image_category = 8
        elif query_image_name in np.arange(900, 1000):
            query_image_category = 9

        k = 0
        for i in range(0, total_retrieved_images):
            retrieved_image_name = int((os.path.splitext(np.array(all_sorted_dists_for_all_quary_images[img][i])[0]))[0])
            if retrieved_image_name in np.arange(0, 100):
                retrieved_image_category = 0
            elif retrieved_image_name in np.arange(100, 200):
                retrieved_image_category = 1
            elif retrieved_image_name in np.arange(200, 300):
                retrieved_image_category = 2
            elif retrieved_image_name in np.arange(300, 400):
                retrieved_image_category = 3
            elif retrieved_image_name in np.arange(400, 500):
                retrieved_image_category = 4
            elif retrieved_image_name in np.arange(500, 600):
                retrieved_image_category = 5
            elif retrieved_image_name in np.arange(600, 700):
                retrieved_image_category = 6
            elif retrieved_image_name in np.arange(700, 800):
                retrieved_image_category = 7
            elif retrieved_image_name in np.arange(800, 900):
                retrieved_image_category = 8
            elif retrieved_image_name in np.arange(900, 1000):
                retrieved_image_category = 9

            if (query_image_category == retrieved_image_category) :
                k += 1

        relevant_retrieved_images = k

        p = relevant_retrieved_images/total_retrieved_images
        r = relevant_retrieved_images/total_relevant_retrieved_images

        precision.append(p)
        recall.append(r)

        img += 1

    average_precision = np.mean(np.array(precision))
    average_recall = np.mean(np.array(recall))

    return print('average_precision :',  average_precision, ',', 'average_recall : ', average_recall)
