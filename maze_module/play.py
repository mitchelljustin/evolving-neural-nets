from pygame.locals import *
import pygame
import random
import time
from maze_module.robot import Robot
from maze_module.maze import Maze
import os
class App:

    windowWidth = 800
    windowHeight = 600
    robot = []
    render = True

    def __init__(self, render=True):
        self._running = True
        self._display_surf = None
        self._image_surfs = [None]*100
        self.robot = [None]*100
        self.startPos = (400, 200)
        self.goalPos = (400, 500)
        self.maze = Maze(800, 600, "./maze_module/maze.png")
        self.robot = Robot(self.maze)
        self.robot.x = self.startPos[0]
        self.robot.y = self.startPos[1]
        self.render = render


    def on_init(self):
        if self.render:
            pygame.init()
            self._display_surf = pygame.display.set_mode((self.windowWidth,self.windowHeight), pygame.HWSURFACE)

            pygame.display.set_caption('Evolved neural nets')
            self._running = True
            self._image_surf = pygame.image.load("./maze_module/robot.png").convert()

    def on_render(self):
        self._display_surf.fill((255,255,255))
        self.maze.draw(self._display_surf)
        self._display_surf.fill((217, 33, 33), ((self.startPos[0]-10,self.startPos[1]-10), (20,20)))
        self._display_surf.fill((66, 217, 33), ((self.goalPos[0]-10,self.goalPos[1]-10), (20,20)))
        self._display_surf.blit(self._image_surf,(self.robot.x,self.robot.y))
        for j in range(len(self.robot.old_places)-1):
            x, y = self.robot.old_places[j]
            x2, y2 = self.robot.old_places[j+1]
            x = round(x)
            y = round(y)
            x2 = round(x2)
            y2 = round(y2)
            pygame.draw.line(self._display_surf, (217,0,0), (x,y), (x2,y2))

        pygame.display.flip()

    def on_cleanup(self):
        if self.render:
            pygame.quit()

    def on_execute(self, neuralNet):
        if self.on_init() == False:
            self._running = False
        step = 0
        while( self._running ):
            step = step + 1
            if self.render:
                pygame.event.pump()
            inputs = self.robot.getSensorValues() + self.robot.getPieValues(self.goalPos[0], self.goalPos[1])
            # print("the inputs are {} ".format(inputs))
            output = neuralNet.activate(inputs)
            # print("the output is {} ".format(output))
            self.robot.rotate(round(output[0]))
            self.robot.move(round(output[1]))
            if step > 400:
                self._running = False
            if self.render:
                if step % 20 == 0:
                    self.on_render()
        self.on_cleanup()
        return 1

if __name__ == "__main__" :
    theApp = App()
    theApp.on_execute()
