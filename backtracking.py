import numpy as np
game = np.array([0,0,4,3,0,0,2,0,9,
                 0,0,5,0,0,9,0,0,1,
                 0,7,0,0,6,0,0,4,3,
                 0,0,6,0,0,2,0,8,7,
                 1,9,0,0,0,7,4,0,0,
                 0,5,0,0,8,3,0,0,0,
                 6,0,0,0,0,0,1,0,5,
                 0,0,3,5,0,8,6,9,0,
                 0,4,2,9,1,0,3,0,0])
game = game.reshape((9,9))

def find_blank(game,n):
    for row in range (9):
        for col in range (9):
            if (game[row][col] == 0):
                n[0] = row
                n[1] = col
                return True
    return False            

def used_row(game,row,num):
    for i in range (9):
        if(game[row][i] == num):
            return False
    return True

def used_col(game,col,num):
    for i in range (9):
        if(game[i][col] == num):
            return False
    return True

def used_box(game,row,col,num):
    row_box = row - (row % 3)
    col_box = col - (col % 3)
    for i in range (3):
        for j in range (3):
            if(game[row_box + i][col_box + j] == num):
                return False
    return True

def validation(game,row,col,num):
    if (used_row(game,row,num) and used_col(game,col,num) and used_box(game,row,col,num)):
        return True
    else:
        return False

def solve(game):
    n = [0,0]

    if(not find_blank(game,n)):
        return True
    
    row = n[0]
    col = n[1]

    for num in range (1, 10):
        if (validation(game,row,col,num)):
            game[row][col] = num
            if (solve(game)):
                return True
            game[row][col] = 0
    return False

if (solve(game)):
    print(game)
else:
    print("This game has no solution!")
