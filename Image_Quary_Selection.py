import os
import numpy as np
from PIL import Image


def image_query_selection(dataset_path, model):

    files = os.listdir(dataset_path)

    selected_imgs = []
    for items in files:
        image = Image.open(os.path.join(dataset_path, items))
        image = image.resize((224, 224))
        image = np.array(image)
        selected_imgs.append(image)
    selected_images = np.array(selected_imgs)

    selected_images_features = model.predict(selected_images)

    query_images_features = selected_images_features
    query_images = selected_images
    query_images_names = files

    return query_images_names, query_images, query_images_features
