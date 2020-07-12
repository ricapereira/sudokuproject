import numpy as np
import os
import keras as k
import tensorflow as tf
import cv2
import random
from keras import layers
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D, MaxPooling2D
from keras.models import Model
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
import h5py

def remove_digit0(train_X,train_Y):
    id = []
    for i in range (train_X.shape[0]):
        if train_Y[i] == 0:
            id.append(i)
    train_Y = np.delete(train_Y, id, 0)
    train_X = np.delete(train_X, id, 0)
    return train_X, train_Y

def get_data(train_X, train_Y, test_X, test_Y):
    train_X, train_Y = remove_digit0(train_X, train_Y)
    test_X, test_Y = remove_digit0(test_X, test_Y)

    train_path = r'Train'  
    list_folder = os.listdir(train_path)
    trset = []
    for folder in list_folder:
        fimages = os.listdir(os.path.join(train_path, folder))
        for f in fimages:
            img = cv2.imread(os.path.join(train_path, folder, f))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, (28,28))
            trset.append(img)
    trset = np.array(trset, dtype = np.uint8)
    trset = trset.reshape(5000, -1)

    train_label = []
    for i in range(0,10):
        temp = 500*[i]
        train_label += temp
    trlabel = np.array(train_label, dtype = np.uint8)


    test_path = r'Test'  
    list_folder = os.listdir(test_path)
    tsset = []
    tslabel = []
    for folder in list_folder:
        fimages = os.listdir(os.path.join(test_path, folder))
        for f in fimages:
            img = cv2.imread(os.path.join(test_path, folder, f))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, (28,28))
            tsset.append(img)
            tslabel.append(int(folder))
    tsset = np.array(tsset, dtype = np.uint8)
    tslabel = np.array(tslabel, dtype = np.uint8)
    tsset = tsset.reshape(3097, -1)

    train_X = np.concatenate((train_X, trset), axis=0)
    train_X = train_X.reshape(-1,28,28,1)
    train_X = train_X / 255
    train_Y = np.concatenate((train_Y, trlabel), axis=0)
    train_Y = tf.keras.utils.to_categorical(train_Y, 10)
    test_X = np.concatenate((test_X, tsset), axis=0)
    test_X = test_X.reshape(-1,28,28,1)
    test_X = test_X / 255
    test_Y = np.concatenate((test_Y, tslabel), axis=0)
    test_Y = tf.keras.utils.to_categorical(test_Y, 10)

    return train_X, train_Y, test_X, test_Y

def shuffle_data(X, Y):
    np.random.seed(3)
    m = X.shape[0]
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:]
    shuffled_Y = Y[permutation,:]

    return shuffled_X, shuffled_Y

def DigitModel(input_shape):

    X_input = Input(input_shape)
    X = Conv2D(6, (5, 5), strides = (1, 1), padding='valid', name = 'conv0')(X_input)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), strides=(2,2), name = 'mpool0')(X)

    X = Conv2D(16, (5, 5), strides = (1, 1), padding='valid', name = 'conv1')(X)
    X = BatchNormalization(axis = 3, name = 'bn1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), strides=(2,2), name = 'mpool1')(X)

    X = Flatten()(X)
    X = Dense(80, activation='relu', name='fc0')(X)
    X = Dense(56, activation='relu', name='fc1')(X)
    X = Dense(10, activation='softmax', name='fc2')(X)

    model = Model(inputs = X_input, outputs = X, name='DigitModel')

    return model


(train_X, train_Y), (test_X, test_Y) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
train_X = train_X.reshape(60000,-1)
test_X = test_X.reshape(10000,-1)

train_X, train_Y, test_X, test_Y = get_data(train_X, train_Y, test_X, test_Y)

train_X, train_Y = shuffle_data(train_X, train_Y)

digitModel = DigitModel(train_X.shape[1:])
digitModel.compile(optimizer = "Adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

EarlyStop_callback = k.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
my_callback=[EarlyStop_callback]

digitModel.fit(x = train_X, y = train_Y, epochs = 34, batch_size = 16, validation_split = 0.1, callbacks = my_callback)

_, acc = digitModel.evaluate(test_X, test_Y, verbose=0)

print('Test Accuracy: ', acc)

Model.save("final_model.h5")







      