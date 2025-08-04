# Project 2
# CS797
# Logan Schraeder

# Based on the LSL-Net

import modules
import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras import layers
from keras.models import Sequential
import os
import pycocotools
import graphviz
import pydot

# Image loading and preprocessing

img_path = os.path.join(os.getcwd(), 'Sample COCO Images', 'helicopter.jpg')
img_height = 416
img_width = 416
# Load image and convert to a tf tensor
input_img = keras.utils.load_img(img_path, target_size=(img_height, img_width))
input_tensor = tf.convert_to_tensor(input_img)
# Add batch dimension to tensor to be compatible with Keras CNN input shapes
input_tensor = np.expand_dims(input_tensor, axis=0)


lsm = modules.lsm()
csp1 = modules.csp()
csp2 = modules.csp()
cspspp = modules.csp_spp()

lsm_out = lsm(input_tensor)
efm_out = modules.efm(lsm_out)
print(efm_out)
# Input into LNB and Accurate Detection Module goes here
# Outputs small/med/large object detection and classification