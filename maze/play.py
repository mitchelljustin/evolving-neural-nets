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
            self.robots[i] = Robot(self.maze)
            self.robots[i].x = self.startPos[0]
            self.robots[i].y = self.startPos[1]


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
        for i in range(1):
            self._display_surf.blit(self._image_surfs[i],(self.robots[i].x,self.robots[i].y))
            for j in range(len(self.robots[i].old_places)-1):
                x, y = self.robots[i].old_places[j]
                x2, y2 = self.robots[i].old_places[j+1]
                x = round(x)
                y = round(y)
                x2 = round(x2)
                y2 = round(y2)
                pygame.draw.line(self._display_surf, (217,0,0), (x,y), (x2,y2))

        pygame.display.flip()

    def on_cleanup(self):
        pygame.quit()

    def on_execute(self, neuralNet):
        if self.on_init() == False:
            self._running = False
        step = 0
        while( self._running ):
            step = step + 1
            pygame.event.pump()
            for i in range(i):
                inputs = self.robots[i].getSensorValues() + self.robots[i].getPieValues(self.goalPos[0], self.goalPos[1])
                output = neuralNet.activate(inputs)
                self.robots[i].rotate(round(output[0]))
                self.robots[i].move(round(output[1]))
            self.on_loop()
            self.on_render()
            if step > 400:
                self._running = False
        self.on_cleanup()

if __name__ == "__main__" :
    theApp = App()
    theApp.on_execute()
