import cv2
import matplotlib.pyplot as plt 
import keras as k
import numpy as np
import operator

path = r'images\\test1.jpg'
img = cv2.imread(path)
img = cv2.resize(img,(700,700))
cv2.imshow('original', img)
cv2.waitKey(0)

def preprocess_img(img):
    #grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Gaussian Blur Filter
    dst = cv2.GaussianBlur(gray,(3,3),3)
    #Transform to inverse binary image
    img = cv2.adaptiveThreshold(dst, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3)
    #Dilate the boundaries
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(2,2))
    img = cv2.dilate(img,kernel,iterations = 1)

    cv2.imshow('dilate', img)
    cv2.waitKey(0)
    return img

img = preprocess_img(img)

def find_corners(img):
    _, contours, h = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    #All points of the largest grid
    largest_grid = contours[0]

    bottom_right, a = max(enumerate([pt[0][0] + pt[0][1] for pt in largest_grid]), key=operator.itemgetter(1))
    top_left, b = min(enumerate([pt[0][0] + pt[0][1] for pt in largest_grid]), key=operator.itemgetter(1))
    bottom_left, c = min(enumerate([pt[0][0] - pt[0][1] for pt in largest_grid]), key=operator.itemgetter(1))
    top_right, d = max(enumerate([pt[0][0] - pt[0][1] for pt in largest_grid]), key=operator.itemgetter(1))

    return [largest_grid[top_left][0], largest_grid[top_right][0], largest_grid[bottom_right][0], largest_grid[bottom_left][0]]

def distance_between(p1,p2):
    x1, x2 = p1[0], p1[1]
    y1, y2 = p2[0], p2[1]
    dist = np.sqrt( (x2 - x1)**2 + (y2 - y1)**2 )
    return dist

corners = find_corners(img)

def cut_and_warp(img, corners):
    top_left, top_right, bottom_right, bottom_left = corners[0], corners[1], corners[2], corners[3]

    corners = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')

    width = max([
            distance_between(bottom_right, bottom_left),
            distance_between(top_right, top_left),
            ])

    height = max([
            distance_between(top_right, bottom_right),
            distance_between(top_left, bottom_left),
            ])

    dst = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype='float32')

    matrix = cv2.getPerspectiveTransform(corners, dst)

    warp = cv2.warpPerspective(img, matrix, (int(width), int(height)))

    cv2.imshow('warp', warp)
    cv2.waitKey(0)
    return warp

warp = cut_and_warp(img, corners)
warp = cv2.GaussianBlur(warp,(3,3),3)
cv2.imshow('final', warp)
cv2.waitKey(0)

frame = cv2.cvtColor(warp, cv2.COLOR_GRAY2RGB)
lines = cv2.HoughLines(img, 1, np.pi /180, 300,0,0)
if (lines is not None):
    lines=lines[0]
    lines = sorted(lines, key=lambda line:line[0])
    pos_h = 0
    pos_v = 0
    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        if (b>0.5):
            if(rho-pos_h>10):
                pos_h = rho
                cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),2)
            else:
                if(rho-pos_v>10):
                    pos_v = rho
                    cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),2)
        
cv2.imshow('lines', frame)
cv2.waitKey(0)