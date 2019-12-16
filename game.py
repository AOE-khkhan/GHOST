# from std lib
import time, json
from random import randint

# third prtylibs
import cv2
import numpy as np
from matplotlib import pyplot as plt

# the moves
UP, DOWN, LEFT, RIGHT, NONE = '__up__', '__down__', '__left__', '__right__', '__none__'

class Game:
    '''grid world game'''

    def __init__(self, start=None, target=None, g=(3, 3), w=30, h=30, event=None, game_play_path='data/game_plays/game_play.json'):
        '''construct Game:
            start -> tuple co-ordinate for starting
            target -> tuple co-ordinate for goal
            g -> grid size
            w -> game width
            h -> game height
        '''

        # start and target
        self.s, self.t = start, target

        # event func
        self.event = lambda :cv2.waitKey(33) if event is None else event

        # set image size and the grid size
        self.w, self.h, self.g = 30, 30, (3, 3)

        # convert grip position to area of coverage
        self.multiplier = lambda x: [int(v*self.g[i]) for i, v in enumerate(x)]

        # update state
        self.update_s = lambda sx, inc: [x+inc[i] if 0 <= x + inc[i] < self.n[i] else x for i, x in enumerate(sx)]

        # get the number of grids
        self.n = list(map(int, [self.h/g[0], self.w/g[1]]))

        # game states
        self.gw = self.sg = self.tg = self.game_state = None

        # color code for Reference
        self.target_code, self.start_code, self.empty_space_code = 125, 255, 0

        # the path to game play
        self.game_play_path = game_play_path

        # get prev saved game plays
        try:
            with open(self.game_play_path, 'r') as f:
                self.game_plays = json.load(f)

        except:
            self.game_plays = {}

    def startGame(self):
        '''initialize a new game'''

        # if game on or off
        self.game_state = True

        # image
        self.gw = np.zeros((self.h, self.w), dtype=np.uint8)

        # first position
        if self.s is None:
            self.s = (randint(0, self.n[0]-1), randint(0, self.n[1]-1))

        if self.t is None:
            self.t = (randint(0, self.n[0]-1), randint(0, self.n[1]-1))
            while self.t == self.s:
                self.t = (randint(0, self.n[0]-1), randint(0, self.n[1]-1))

        # target grid
        self.tg = self.multiplier(self.t)

        # set the target
        self.gw[self.tg[0]: self.tg[0]+self.g[0], self.tg[1]:self.tg[1] + self.g[1]] = self.target_code
        
        # resize the window
        # cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('frame', 300, 300)

        # the last state and last state grid
        self.last_s = self.last_sg = None

        # the game  game_signature
        self.game_signature = " ".join(list(map(str, self.s))) + " " + " ".join(list(map(str, self.t)))

        # if the start-target state does not exist
        if self.game_signature not in self.game_plays:
            self.game_plays[self.game_signature] = []

        # the way the game was played
        self.game_play = []

        return

    def moveUp(self):
        self.game_play.append(UP)
        self.s = self.update_s(self.s, [-1, 0])

    def moveDown(self):
        self.game_play.append(DOWN)
        self.s = self.update_s(self.s, [1, 0])

    def moveLeft(self):
        self.game_play.append(LEFT)
        self.s = self.update_s(self.s, [0, -1])

    def moveRight(self):
        self.game_play.append(RIGHT)
        self.s = self.update_s(self.s, [0, 1])

    def controller(self, k):
        if k == 27:    # Esc key to stop
            print('Game Quit!')
            self.game_state = False

        elif self.sg == self.tg:
            print('Game Solved!')
            
            gp = tuple(self.game_play)
            if gp in self.game_plays[self.game_signature]:
                return
            
            self.game_plays[self.game_signature].append(gp)

            with open(self.game_play_path, 'w') as f:
                json.dump(self.game_plays, f)

            self.game_state = False

        elif k == 119:
            self.moveUp()
        
        elif k == 97:
            self.moveLeft()

        elif k == 115:
            self.moveRight()

        elif k == 122:
            self.moveDown()
        
        else:
            return

        return

    def update(self):
        if self.last_s == self.s:
            return

        # start grid
        self.sg = self.multiplier(self.s)

        # update the game world
        if self.last_s != self.s:
            if self.last_sg is not None:
                self.gw[self.last_sg[0]: self.last_sg[0]+self.g[0], self.last_sg[1]:self.last_sg[1] + self.g[1]] = self.empty_space_code
            self.gw[self.sg[0]: self.sg[0]+self.g[0], self.sg[1]:self.sg[1] + self.g[1]] = self.start_code
        
        # update last state
        self.last_s, self.last_sg = self.s, self.sg
        return

    def runDiscretely(self, k=-1):
        # start the game
        if not self.game_state:
            self.startGame()

        # update the game state
        self.update()
        
        # controls the game
        self.controller(k)
    
    def __desc__(self):
        # close all open windows
        cv2.destroyAllWindows()

    def runContinously(self):
        # start the game
        if not self.game_state:
            self.startGame()

        while self.game_state:
            # update the game state
            self.update()

            # render
            cv2.imshow('frame', self.gw)

            # controls the game
            self.controller(self.event())

        # close all open windows
        cv2.destroyAllWindows()

def main():
    # the start and end grid
    start, target = (3, 3), (7, 7)

    # get game instance
    game_object = Game(start=start, target=target)

    # run the game
    game_object.runContinously()

if __name__ == '__main__':
    main()
