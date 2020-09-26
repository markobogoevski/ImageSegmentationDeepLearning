#!/usr/bin/env python
# coding: utf-8

# In[2]:


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
    
parser = argparse.ArgumentParser()
parser.add_argument('--image_path', help='Choose the path of the image you wish to segment.', type=str, required=True)
args = parser.parse_args()

model_path="trained_models/SegNet_VGG16base_CamVid_20epochs.h5"
dataset_path="CamVid"
num_classes=32

# load the model
model = load_model(model_path, compile=False)
model.compile(
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4),
loss = categorical_crossentropy_with_logits,
metrics = [MeanIoU(num_classes)]) # num classes is argument

class_dict_path = 'CamVid/class_dict.csv'
image_to_predict = cv2.cvtColor(cv2.imread(args.image_path), cv2.COLOR_BGR2RGB)
pred = predict_image(model, args.image_path, dataset_path, num_classes=num_classes, target_size=(256, 256))

fig = plt.figure(figsize=(15,15))

plt.subplot(1,2,1)
plt.title('Original image')
plt.imshow(image_to_predict)

plt.subplot(1,2,2)
plt.title('Segmented image')
plt.imshow(pred)

