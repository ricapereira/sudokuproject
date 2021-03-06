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
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def getdata():
    bigtrainpath = r'Train\big'
    smalltrainpath = r'Train\small'
    testpath = r'Test'
    paths = (bigtrainpath, smalltrainpath, testpath)
    trset = []
    trlabel = []
    tsset = []
    tslabel = []
    for path in paths:
        a = 0
        list_folder = os.listdir(path)
        for folder in list_folder:
            a = a+1
            fimages = os.listdir(path)
            img = cv2.imread(os.path.join(path, folder))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if (path == bigtrainpath):
                for i in range(1,501):
                    for j in range(1,51):
                        data = img[(i-1)*28:i*28, (j-1)*28:j*28]
                        data.reshape(28,28,1)
                        trset.append(data)
                        trlabel.append(a)
            elif (path == smalltrainpath):
                for k in range(1,101):
                    for l in range(1,51):
                        data = img[(k-1)*28:k*28, (l-1)*28:l*28]
                        data.reshape(28,28,1)
                        trset.append(data)
                        trlabel.append(a)
            elif (path == testpath):
                for m in range(1,51):
                    for n in range(1,51):
                        data = img[(m-1)*28:m*28, (n-1)*28:n*28]
                        data.reshape(28,28,1)
                        tsset.append(data)
                        tslabel.append(a)
    
    trset = np.array(trset, dtype = np.uint8)
    trlabel = np.array(trlabel, dtype = np.uint8)
    trsetextra = np.zeros((30000,28,28))
    trlabelextra = np.zeros((30000,))
    train_X = np.concatenate((trset, trsetextra), axis=0)
    train_Y = np.concatenate((trlabel, trlabelextra), axis=0)
    print(train_X.shape)
    train_X = train_X[:,:,:,np.newaxis]
    print(train_X.shape)
    train_Y = tf.keras.utils.to_categorical(train_Y, 10)

    tsset = np.array(tsset, dtype = np.uint8)
    tslabel = np.array(tslabel, dtype = np.uint8)
    tssetextra = np.zeros((2500,28,28))
    tslabelextra = np.zeros((2500,))
    test_X = np.concatenate((tsset, tssetextra), axis=0)
    test_Y = np.concatenate((tslabel, tslabelextra), axis=0)
    print(test_X.shape)
    test_X = test_X[:,:,:,np.newaxis]
    print(test_X.shape)
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


train_X, train_Y, test_X, test_Y = getdata()

train_X, train_Y = shuffle_data(train_X, train_Y)
test_X, test_Y = shuffle_data(test_X, test_Y)

digitModel = DigitModel(train_X.shape[1:])

digitModel.compile(optimizer = "Adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

EarlyStop_callback = k.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
my_callback=[EarlyStop_callback]

digitModel.fit(x = train_X, y = train_Y, epochs = 34, batch_size = 16, validation_split = 0.1, callbacks = my_callback)

_, acc = digitModel.evaluate(test_X, test_Y, verbose=0)

print('Test Accuracy: ', acc)

digitModel.save(r'newmodel.h5')

