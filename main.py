import numpy as np
import os
import keras as k
import tensorflow as tf
import cv2
import random
from keras import layers
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D, MaxPooling2D
from keras.models import Model, load_model
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
import h5py
import preprocessing
import backtracking

digitModel = load_model(r'final_model.h5', compile=False)

img_path = r'images\\test1.jpg'
digits = preprocessing.img_to_digits(img_path)
cv2.imshow('digit1',digits[1])
cv2.waitKey(0)
#digit = digitModel.predict_classes()
