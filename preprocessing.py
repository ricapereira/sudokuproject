import cv2
import matplotlib.pyplot as plt 
import keras as k
import numpy as np

path = r'images\\test1.jpg'
img = cv2.imread(path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', img)
cv2.waitKey(0)
