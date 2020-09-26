#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from models.network import Network
from models.segnet import SegNet
from utils.utils import *
from utils.callbacks import *
from utils.data_generator import *
from utils.layers import *
from utils.learning_rate import *
from utils.losses import *
from utils.optimizers import *
from utils.metrics import *
from utils.helpers import *
layers = tf.keras.layers
from tensorflow.keras.models import load_model
import argparse
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def predict_image(model, image_path, dataset, num_classes, color_encoded=True, target_size=(256,256), base_model=None):
    """
    Function which tests the provided model on only one image
    Parameters:
    model: the model built by the builder function which you want to use, tf.Model
    image_path: the path of the image to be segmented, str
    dataset: path to the dataset, str
    num_classes: number of classes in the dataset, int
    color_encoded: whether to color encode the resulting segmented image according to the class_dict file
    base_model: the base model, string or None
    """

    # check the image path in drive
    if not os.path.exists(image_path):
        raise ValueError('The path \'{}\' does not exist the image file.'.format(image_path))

    if base_model is None:
        base_model='VGG16'
    # begin testing
    print("\n***** Begin testing *****")
    print("Model -->", model)
    print("Base Model -->", base_model)
    print("Num Classes -->", num_classes)
    print("")

    # load_images
    image_names=list()
    if os.path.isfile(image_path):
        image_names.append(image_path)
    else:
        for f in os.listdir(image_path):
            image_names.append(os.path.join(image_path, f))
        image_names.sort()

   
    csv_color_file = os.path.join(dataset, 'class_dict.csv')
    _, color_values = get_colored_info(csv_color_file)

    for i, name in enumerate(image_names):
        sys.stdout.write('\rRunning test image %d / %d'%(i+1, len(image_names)))
        sys.stdout.flush()

        image = cv2.resize(load_image(name),
                          dsize=target_size)
        image = imagenet_utils.preprocess_input(image.astype(np.float32), data_format='channels_last', mode='torch')

        # image processing
        if np.ndim(image) == 3:
            image = np.expand_dims(image, axis=0)
        assert np.ndim(image) == 4

        # get the prediction
        prediction = model.predict(image)

        if np.ndim(prediction) == 4:
            prediction = np.squeeze(prediction, axis=0)

        # decode one-hot
        prediction = decode_one_hot(prediction)

        # color encode
        if color_encoded:
            prediction = color_encode(prediction, color_values)

        # get PIL file
        prediction = Image.fromarray(np.uint8(prediction))

        # save the prediction to drive
        # _, file_name = os.path.split(name)
        # prediction.save(os.path.join(paths['prediction_path'], file_name))
        return prediction

