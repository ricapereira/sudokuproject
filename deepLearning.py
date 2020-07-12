import numpy as np
import os
import keras as k
import tensorflow as tf
import cv2

(train_X, train_Y), (test_X, test_Y) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
train_X = train_X.reshape(60000,-1)
test_X = test_X.reshape(10000,-1)
print(test_X.shape)
print(test_Y.shape)

def remove_digit0(train_X,train_Y):
    id = []
    for i in range (train_X.shape[0]):
        if train_Y[i] == 0:
            id.append(i)
    train_Y = np.delete(train_Y, id, 0)
    train_X = np.delete(train_X, id, 0)
    return train_X, train_Y

train_X, train_Y = remove_digit0(train_X, train_Y)
test_X, test_Y = remove_digit0(test_X, test_Y)
print(train_Y.shape)
print(train_X.shape)
print(test_Y.shape)
print(test_X.shape)

train_path = r'Train'  
list_folder = os.listdir(train_path)

for folder in list_folder:
    fimages = os.listdir(os.path.join(train_path, folder))
    for f in fimages:
        img = cv2.imread(os.path.join(train_path, folder, f))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (28,28))
        trset = np.concatenate((img), axis=0)
    print(trset.shape)
train_label = []
for i in range(0,10):
    temp = 500*[i]
    train_label += temp

      