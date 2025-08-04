# EfficientDet Inference

# References:
# https://www.kaggle.com/models/tensorflow/efficientdet
# https://www.kaggle.com/datasets/dasmehdixtr/drone-dataset-uav/

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
import os
import numpy as np

# Load image dataset
img_path = os.path.join(os.getcwd(), 'drone_dataset_xml_format', 'drone_jpgs')
files = os.listdir(img_path)
img_height = 416
img_width = 416

# Load EfficientDet d0v1
detector = hub.load("https://kaggle.com/models/tensorflow/efficientdet/frameworks/TensorFlow2/variations/d0/versions/1")

for img in files:
    # Load image and convert to a tf tensor
    path = os.path.join(img_path, img)
    input_img = keras.utils.load_img(path, target_size=(img_height, img_width))
    input_tensor = tf.convert_to_tensor(input_img)
    # Add batch dimension to tensor to be compatible with Keras CNN input shapes
    input_tensor = np.expand_dims(input_tensor, axis=0)
    output = detector(input_tensor)
    output_class = output["detection_classes"]
    print(output_class)
