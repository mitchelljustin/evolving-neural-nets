import numpy as np
import pygame
import random
from scipy.spatial import distance

from maze_module.maze import Maze
from maze_module.robot import Robot

END_STEP_NUM = 400
BEHAVIOUR_SAMPLE_STEP_NUMS = [
    (i + 1) * END_STEP_NUM // 3 - 1
    for i in range(3)
]


class App:
    windowWidth = 800
    windowHeight = 600
    render = True
    colors = [
        (217, 0, 0),
        (0, 217, 0),
        (0, 0, 217),
        (217, 217, 0),
        (217, 0, 217)
    ]

    def __init__(self, render=True):
        self._running = True
        self._display_surf = None
        self._image_surfs = [None] * 100
        self.goalPos = (400, 500)
        self.startPos = (400, 200)
        self.maze = Maze(800, 600, "./maze_module/maze.png")
        self.render = render

    def on_init(self):
        if self.render:
            pygame.init()
            self._display_surf = pygame.display.set_mode((self.windowWidth, self.windowHeight), pygame.HWSURFACE)

            pygame.display.set_caption('Evolved neural nets')
            self._running = True
            self._image_surf = pygame.image.load("./maze_module/robot.png").convert()

    def on_render(self):
        self._display_surf.fill((255, 255, 255))
        self.maze.draw(self._display_surf)
        self._display_surf.fill((217, 33, 33), ((self.startPos[0] - 10, self.startPos[1] - 10), (20, 20)))
        self._display_surf.fill((66, 217, 33), ((self.goalPos[0] - 10, self.goalPos[1] - 10), (20, 20)))
        for robot in self.robots:
            self._display_surf.blit(self._image_surf, (robot.x, robot.y))
            for j in range(len(robot.old_places) - 1):
                x, y = robot.old_places[j]
                x2, y2 = robot.old_places[j + 1]
                x = round(x)
                y = round(y)
                x2 = round(x2)
                y2 = round(y2)
                pygame.draw.line(self._display_surf, robot.color, (x, y), (x2, y2))

        pygame.display.flip()


    def reset(self):
        self.robots = []

    def create_n_robots(self, neuralNets):
        robots = []
        for i in range(len(neuralNets)):
            robots.append(Robot(self.maze, neuralNets[i], self.colors[i%5]))


    def execute(self, neuralNets):
        if not self.on_init():
            self._running = False
        behaviour = []
        results = []
        self.robots = self.create_n_robots(neuralNets)
        for step in range(END_STEP_NUM):
            for robot in self.robots:
                if self.render:
                    pygame.event.pump()
                inputs = robot.getSensorValues() + robot.getPieValues(*self.goalPos)
                left_right, fwd_back = robot.neuralNet.activate(inputs)
                robot.rotate(round(left_right))
                robot.move(round(fwd_back))
                if step in BEHAVIOUR_SAMPLE_STEP_NUMS:
                    behaviour.append((self.robot.x, self.robot.y))
                if self.render and step % 30 == 0:
                    self.on_render()
        for robot in self.robots:
            robot_pos = (robot.x, robot.y)
            objective_value = 640 - distance.euclidean(robot_pos, self.goalPos)
            behaviour = np.array(behaviour).reshape([len(BEHAVIOUR_SAMPLE_STEP_NUMS) * 2])
            results.append(np.concatenate([behaviour, [objective_value]], -1))
        self.reste()
        return results
