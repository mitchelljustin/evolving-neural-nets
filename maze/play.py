from pygame.locals import *
import pygame
import random
from robot import Robot
from maze import Maze
class App:

    windowWidth = 800
    windowHeight = 600
    robots = []

    def __init__(self):
        self._running = True
        self._display_surf = None
        self._image_surfs = [None]*100
        self.robots = [None]*100
        self.startPos = (400, 200)
        self.goalPos = (400, 500)
        self.maze = Maze(800, 600, "maze.png")
        for i in range(100):
            self.robots[i] = Robot()
            self.robots[i].x = random.choice(range(700))
            self.robots[i].y = random.choice(range(500))


    def on_init(self):
        pygame.init()
        self._display_surf = pygame.display.set_mode((self.windowWidth,self.windowHeight), pygame.HWSURFACE)

        pygame.display.set_caption('Evolved neural nets')
        self._running = True
        for i in range(len(self.robots)):
            self._image_surfs[i] = pygame.image.load("robot.png").convert()

    def on_event(self, event):
        if event.type == QUIT:
            self._running = False

    def on_loop(self):
        pass

    def on_render(self):
        self._display_surf.fill((255,255,255))
        self.maze.draw(self._display_surf)
        self._display_surf.fill((217, 33, 33), ((self.startPos[0]-10,self.startPos[1]-10), (20,20)))
        self._display_surf.fill((66, 217, 33), ((self.goalPos[0]-10,self.goalPos[1]-10), (20,20)))
        # for i in range(len(self.robots)):
            # pygame.transform.rotate(self._image_surfs[i], self.robots[i]._rotation)
            # self._display_surf.blit(self._image_surfs[i],(self.robots[i].x,self.robots[i].y))
        pygame.display.flip()

    def on_cleanup(self):
        pygame.quit()

    def on_execute(self):
        if self.on_init() == False:
            self._running = False

        while( self._running ):
            pygame.event.pump()
            for i in range(len(self.robots)):
                r = random.choice(range(4))
                self.robots[i].rotate(r)
                if (r == 0):
                    self.robots[i].moveRight()

                if (r == 1):
                    self.robots[i].moveLeft()

                if (r == 2):
                    self.robots[i].moveUp()

                if (r == 3):
                    self.robots[i].moveDown()
            self.on_loop()
            self.on_render()
        self.on_cleanup()

if __name__ == "__main__" :
    theApp = App()
    theApp.on_execute()
