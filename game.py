import cv2
import numpy as np
from random import randint

# set image size and the grid size
w, h, g = 30, 30, (3, 3)

# convert grip position to area of coverage
multiplier = lambda x: [int(v*g[i]) for i, v in enumerate(x)]

# get the number of grids
n = list(map(int, [h/g[0], w/g[1]]))

# resize the window
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cv2.resizeWindow('frame', 300, 300)

# image
gw = np.zeros((h, w), dtype=np.uint8)

# first position
s = (randint(0, n[0]-1), randint(0, n[1]-1))

t = (randint(0, n[0]-1), randint(0, n[1]-1))
while t == s:
    t = (randint(0, n[0]-1), randint(0, n[1]-1))

# target grid
tg = multiplier(t)

# set the target
gw[tg[0]: tg[0]+g[0], tg[1]:tg[1] + g[1]] = 255

# last state
last_s = last_sg = None

# update state
update_s = lambda sx, inc: [x+inc[i] if 0 <= x + inc[i] < n[i] else x for i, x in enumerate(sx)]
    
while True:
    # start grid
    sg = multiplier(s)

    # update the game world
    if last_s != s:
        if last_sg is not None:
            gw[last_sg[0]: last_sg[0]+g[0], last_sg[1]:last_sg[1] + g[1]] = 0
        gw[sg[0]: sg[0]+g[0], sg[1]:sg[1] + g[1]] = 125

    # render
    cv2.imshow('frame', gw)

    # update last state
    last_s = s
    last_sg = sg

    # get keycode
    k = cv2.waitKey(33)

    if k == 27:    # Esc key to stop
        print('Game Quit!')
        break
    
    elif sg == tg:
        print('Game Solved!')
        break

    elif k == -1:  # normally -1 returned,so don't print it
        continue
    
    elif k == 119:
        s = update_s(s, [-1, 0])
    
    elif k == 97:
        s = update_s(s, [0, -1])

    elif k == 115:
        s = update_s(s, [0, 1])

    elif k == 122:
        s = update_s(s, [1, 0])

cv2.destroyAllWindows()
