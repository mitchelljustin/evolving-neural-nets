from robot import Robot
from maze_module.maze import Maze


class Game():
    _maze = None
    robots = []

    def __init__(self, maze):

        # do something to the image
        self._maze = maze
        for i in range(200):
            robots[i] = Robot()

