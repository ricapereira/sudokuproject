import os
import cv2
import numpy as np
import keras as k

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
    
    trset = np.array(trset, dtype = np.uint8)
    trlabel = np.array(trlabel, dtype = np.uint8)
    trsetextra = np.zeros((30000,28,28))
    trlabelextra = np.zeros((30000,))
    train_X = np.concatenate((trset, trsetextra), axis=0)
    train_Y = np.concatenate((trlabel, trlabelextra), axis=0)
    #train_X.reshape(30000,28,28,1)
    train_Y = tf.keras.utils.to_categorical(train_Y, 10)


    print(train_X.shape)
    print(train_Y.shape)

    return train_X, train_Y

trset, trlabel = getdata()

