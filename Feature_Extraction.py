import os
import numpy as np
from keras.applications.resnet50 import ResNet50, preprocess_input
from PIL import Image
from keras.models import Model
import h5py


def feature_extraction(dataset_path, feature_extracted_path):

    # Read data
    # images = []
    # for imgs in os.listdir(dataset_path):
    #     image = Image.open(os.path.join(dataset_path, imgs))
    #     image = image.resize((224, 224))
    #     image = np.array(image)
    #     images.append(image)
    #
    # images = np.array(images)
    # np.random.shuffle(images)

    # images = preprocess_input(images)

    # Load model
    model = ResNet50()

    # remove the output layer
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

    # features = model.predict(images)

    # Write data
    # features_hf = h5py.File(feature_extracted_path, 'w')
    # features_hf.create_dataset('features', data=features)
    # features_hf.close()

    features = h5py.File(feature_extracted_path, 'r')
    features = np.array(features.get('features')).astype(np.float32)

    return model, features
