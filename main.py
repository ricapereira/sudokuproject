import os
import keras as k
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
import h5py as h5
import preprocessing
import backtracking
import numpy as np
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

#model.h5 -> largest dataset (64000)
digitModel = load_model(r'model.h5', compile=False)

img_path = r'images\\test2.jpg'
digits = preprocessing.img_to_digits(img_path)
'''
for d in range(81):
    cv2.imshow('num'+str(d), digits[d])
    cv2.waitKey(0)
'''
digit = digitModel.predict(digits)


sudoku = digit.argmax(axis=-1)
sudoku = sudoku.reshape(9,9)
sudoku = np.transpose(sudoku)
print(sudoku)
