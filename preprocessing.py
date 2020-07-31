import cv2
import keras as k
import numpy as np
import operator

def preprocess_img(img):
    #grayscale
    #gray = cv2.resize(img, (1000,1000))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Gaussian Blur Filter
    dsst = cv2.GaussianBlur(gray,(9,9),0)
    #Transform to inverse binary image
    img = cv2.adaptiveThreshold(dsst, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5)
    return img

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

    return warp, corners, dst

def make_grid(warp):
    side = min(warp.shape[0],warp.shape[1])
    side = side / 9
    squares = []
    frame = cv2.cvtColor(warp, cv2.COLOR_GRAY2RGB)
    for i in range(9):
        for j in range(9):
            tl = (i*side, j*side)
            br = ((i+1)*side, (j+1)*side)
            squares.append((tl,br))
    for square in squares:
        frame = cv2.rectangle(frame, tuple(int(x) for x in square[0]), tuple(int(x) for x in square[1]), (0,0,255))
    return squares, frame

def center_pad(num):
    height, width = num.shape
    if height % 2 == 0:
        top = int((28-height) / 2)
        bottom = top
    else:
        top = int((28-height) / 2)
        bottom = top + 1
    if width % 2 == 0:
        right = int((28-width) / 2)
        left = right
    else:
        right = int((28-width) / 2)
        left = right + 1
    num = np.pad(num, ((top,bottom), (left, right)), 'constant')
    return num

def cut_square(frame, square, warp):
    return warp[int(square[0][1]):int(square[1][1]),int(square[0][0]):int(square[1][0])]

def extract_digits(frame, squares, warp):
    num = []
    for i in range(81):
        num.append(cut_square(frame, squares[i], warp))
        num[i] = cv2.resize(num[i], (28,28))
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(1,1))
        num[i] = cv2.morphologyEx(num[i], cv2.MORPH_DILATE, kernel, iterations=1)
        h, w = num[i].shape
        min_area = 0.01*h*w
        max_area = 0.8*h*w
        cnts = cv2.findContours(num[i], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts = [c for c in cnts if max_area>cv2.contourArea(c)>min_area]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        try:
            x,y,w,h = cv2.boundingRect(cnts[0])
            if np.all((num[i][11:16,11:16] == 0)) == True:
                num[i] = np.zeros((28,28,1))
            else:
                if (x,y,w,h) == (0,0,28,28) or ((x,w) == (0,28) and (h-y)<13) or ((y,h) == (0,28) and (w-x)<10):
                    num[i] = np.zeros((28,28,1))
                else:
                    num[i] = num[i][y:h+8, x:w+8]#8[y:h+6, x:w+6]
                    num[i] = center_pad(num[i])
                    num[i] = num[i].reshape(28,28,1)
            
        except IndexError:
            num[i] = np.zeros((28,28,1))

    num = np.array(num, dtype = np.uint8)
    return num

def img_to_digits(path):
    img0 = cv2.imread(path)
    if (img0.shape[0]<600) or (img0.shape[1]<600):
        img0 = cv2.resize(img0,(600,600))
    if (img0.shape[0]>1100) or (img0.shape[1]>1100):
        img0 = cv2.resize(img0,(850,850))
    img = preprocess_img(img0)
    corners = find_corners(img)
    warp, corners, dst = cut_and_warp(img, corners)
    pic = cv2.GaussianBlur(warp,(3,3),3)
    pic = houghtransf(pic)
    pic = cv2.cvtColor(pic, cv2.COLOR_RGB2GRAY)
    squares, frame = make_grid(pic)
    digits = extract_digits(frame, squares, pic)

    return digits, pic, img0, corners, dst

def houghtransf(binary_image):
    edges = cv2.Canny(binary_image, 50, 200)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=10, maxLineGap=250)
    frame = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2RGB)
    for l in lines:
        x1,y1,x2,y2 = l[0]
        cv2.line(frame,(x1,y1),(x2,y2),(0,0,0),10, cv2.LINE_AA)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(2,2))
    frame = cv2.morphologyEx(frame, cv2.MORPH_ERODE, kernel, iterations=1)
    frame = cv2.dilate(frame,kernel,iterations = 1)
    frame = cv2.GaussianBlur(frame,(5,5),0)
    #cv2.imshow('final', frame)
    #cv2.waitKey(0)
    return frame    

def print_final(dst,corners,warp,img0):
    M = cv2.getPerspectiveTransform(dst,corners)
    
    img = cv2.warpPerspective(warp,M,(img0.shape[1],img0.shape[0]))
    img = cv2.bitwise_not(img)
    img = cv2.bitwise_and(img,img0)
    return img






