import os
import keras as k
import cv2
from keras.models import load_model
import h5py as h5
import preprocessing
import backtracking
import numpy as np


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

#model.h5 -> old dataset (64.000)
#newmodel.h5 -> new dataset (300.000)
digitModel = load_model(r'newmodel.h5', compile=False)

img_path = r'images\\test3.jpg'
digits = preprocessing.img_to_digits(img_path)

digit = digitModel.predict(digits)

sudoku = digit.argmax(axis=-1)
sudoku = sudoku.reshape(9,9)
sudoku = np.transpose(sudoku)
print()
print('Your Game:')
print()
print(sudoku)
print()

if (backtracking.solve(sudoku)):
        print('Solved Game:')
        print()
        print(sudoku)
else:
    print("This game has no solution!")
