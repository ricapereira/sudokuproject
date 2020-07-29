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

img_path = r'images\\test6.jpg'
digits, warp, img0, corners, dst = preprocessing.img_to_digits(img_path)
warp = cv2.cvtColor(warp, cv2.COLOR_GRAY2RGB)
cv2.imshow('original', img0)
cv2.waitKey(0)
'''
for i in range(81):
        cv2.imshow('num'+str(i), digits[i])
        cv2.waitKey(0)
'''
digit = digitModel.predict(digits)

sudoku = digit.argmax(axis=-1)
sudoku = sudoku.reshape(9,9)
sudoku = np.transpose(sudoku)
solved = np.copy(sudoku)
grid = cv2.imread(r'images\\grid\\grid.jpg')
side = int(grid.shape[0]/9)
correction = int(grid.shape[0]/33)
for i in range(9):
        for j in range(9):
                if sudoku[i][j] != 0:
                        x = int((j*side)+(side/2))-correction
                        y = int((i*side)+(side/2))+correction
                        cv2.putText(grid, str(sudoku[i][j]), (x,y), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0,0,0), 4)
print()
cv2.imshow('check', grid)
cv2.waitKey(0)
warp = np.zeros(warp.shape, np.uint8)
sidex = int(warp.shape[1]/9)
sidey = int(warp.shape[0]/9)
correctionx = int(warp.shape[1]/33)
correctiony = int(warp.shape[0]/33)
while True:
        ans = input('Todos os digitos da imagem estão corretos? (Y para sim / N para não)')

        if ans in ['y', 'Y', 'sim', 'Sim', 'SIM']:
                if (backtracking.solve(solved)):
                        for m in range(9):
                                for n in range(9):
                                        if sudoku[m][n] == 0:
                                                xx = int((n*sidex)+(sidex/2))-correctionx
                                                yy = int((m*sidey)+(sidey/2))+correctiony
                                                cv2.putText(warp, str(solved[m][n]), (xx,yy), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255,255,0), 3)
                        final = preprocessing.print_final(dst, corners, warp, img0)
                        cv2.imshow('final', final)
                        cv2.waitKey(0)
                        break
                else:
                        print("This game has no solution!")
                        break
        elif ans in ['n', 'N', 'nao', 'Não', 'não', 'Nao', 'NAO', 'NÃO']:
                print('Tire outra foto!')
                break
        else:
                continue
        

