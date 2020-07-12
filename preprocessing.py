import cv2
import matplotlib.pyplot as plt 
import keras as k
import numpy as np
import operator

path = r'images\\test3.jpg'
img0 = cv2.imread(path)
if (img0.shape[0]<600) or (img0.shape[1]<600):
    img0 = cv2.resize(img0,(600,600))
cv2.imshow('original', img0)
cv2.waitKey(0)

def preprocess_img(img):
    #grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Gaussian Blur Filter
    dst = cv2.GaussianBlur(gray,(3,3),3)
    #Transform to inverse binary image
    img = cv2.adaptiveThreshold(dst, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3)
    #Dilate the boundaries
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    img = cv2.dilate(img,kernel,iterations = 1)

    cv2.imshow('dilate', img)
    cv2.waitKey(0)
    return img

img = preprocess_img(img0)

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
'''
def houghtransf(warp):
    lines = cv2.HoughLinesP(warp,2,np.pi/2,500,minLineLength=450,maxLineGap=10)
    frame = cv2.cvtColor(warp, cv2.COLOR_GRAY2RGB)
    print(lines.shape)
    for l in lines:
        x1,y1,x2,y2 = l[0]
        cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),2, cv2.LINE_AA)
    return frame
 

lines = houghtransf(warp)
cv2.imshow('lines', lines)
cv2.waitKey(0)
'''        
edges = cv2.Canny(warp, 90, 150, apertureSize = 3)
frame = cv2.cvtColor(warp, cv2.COLOR_GRAY2RGB)

lines = cv2.HoughLines(edges,1,np.pi/180,150)

if not lines.any():
    print('No lines were found')
    exit()

if filter:
    rho_threshold = 15
    theta_threshold = 0.1

    # how many lines are similar to a given one
    similar_lines = {i : [] for i in range(len(lines))}
    for i in range(len(lines)):
        for j in range(len(lines)):
            if i == j:
                continue

            rho_i,theta_i = lines[i][0]
            rho_j,theta_j = lines[j][0]
            if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
                similar_lines[i].append(j)

    # ordering the INDECES of the lines by how many are similar to them
    indices = [i for i in range(len(lines))]
    indices.sort(key=lambda x : len(similar_lines[x]))

    # line flags is the base for the filtering
    line_flags = len(lines)*[True]
    for i in range(len(lines) - 1):
        if not line_flags[indices[i]]: # if we already disregarded the ith element in the ordered list then we don't care (we will not delete anything based on it and we will never reconsider using this line again)
            continue

        for j in range(i + 1, len(lines)): # we are only considering those elements that had less similar line
            if not line_flags[indices[j]]: # and only if we have not disregarded them already
                continue

            rho_i,theta_i = lines[indices[i]][0]
            rho_j,theta_j = lines[indices[j]][0]
            if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
                line_flags[indices[j]] = False # if it is similar and have not been disregarded yet then drop it now

print('number of Hough lines:', len(lines))

filtered_lines = []

if filter:
    for i in range(len(lines)): # filtering
        if line_flags[i]:
            filtered_lines.append(lines[i])

    print('Number of filtered lines:', len(filtered_lines))
else:
    filtered_lines = lines

for line in filtered_lines:
    rho,theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),2)


cv2.imshow('lines', frame)
cv2.waitKey(0)

